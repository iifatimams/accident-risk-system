package ae.aus.roadsense

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.UUID
import kotlin.math.sqrt

class TelemetryService : Service(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private var wakeLock: PowerManager.WakeLock? = null
    private val obdManager = ObdManager()
    private val httpClient = OkHttpClient()
    private val handler = Handler(Looper.getMainLooper())

    private val targetNgrokUrl = "https://evasion-stung-makeover.ngrok-free.dev"

    private var currentLat: Double = 0.0
    private var currentLon: Double = 0.0
    private var accelMag: Double = 0.0
    private var gyroMag: Double = 0.0
    private var isStreaming = false

    @Volatile private var liveHeartRate: Int = 0
    @Volatile private var isWhoopConnected = false
    private var bluetoothGatt: BluetoothGatt? = null
    private var bluetoothAdapter: BluetoothAdapter? = null
    private var isScanning = false
    private var obdWorkerThread: Thread? = null

    companion object {
        private const val CHANNEL_ID = "RoadSenseTelemetryChannel"
        private const val NOTIFICATION_ID = 1001

        private val HR_SERVICE_UUID = UUID.fromString("0000180d-0000-1000-8000-00805f9b34fb")
        private val HR_MEASUREMENT_UUID = UUID.fromString("00002a37-0000-1000-8000-00805f9b34fb")
        private val CLIENT_CONFIG_DESCRIPTOR = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundServiceNotification()
        acquireWakeLock()

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager

        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME)

        initializeLocation()
        startObdPollingThread()
        startStreamingLoop()
    }

    @SuppressLint("MissingPermission")
    private fun initializeLocation() {
        try {
            val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            val lastNet = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            val bestLocation = lastGps ?: lastNet

            bestLocation?.let {
                currentLat = it.latitude
                currentLon = it.longitude
            }

            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, this)
            locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 1000L, 0f, this)
        } catch (e: Exception) {
            Log.e("TelemetryService", "Location error: ${e.message}")
        }
    }

    private fun startObdPollingThread() {
        isStreaming = true
        obdWorkerThread = Thread {
            while (isStreaming) {
                if (obdManager.isConnected) {
                    obdManager.pollObdData()
                    try { Thread.sleep(60) } catch (e: Exception) {}
                } else {
                    ensureObdConnected()
                    try { Thread.sleep(2000) } catch (e: Exception) {}
                }
            }
        }.apply { start() }
    }

    @SuppressLint("MissingPermission")
    private fun ensureWhoopConnected() {
        if (isWhoopConnected || isScanning) return

        try {
            val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
            bluetoothAdapter = bluetoothManager.adapter ?: return

            val bondedWhoop = bluetoothAdapter?.bondedDevices?.firstOrNull { device ->
                val name = device.name?.uppercase() ?: ""
                name.contains("WHOOP")
            }

            if (bondedWhoop != null) {
                bluetoothGatt = bondedWhoop.connectGatt(
                    this@TelemetryService,
                    false,
                    gattCallback,
                    BluetoothDevice.TRANSPORT_LE
                )
                return
            }

            val scanner = bluetoothAdapter?.bluetoothLeScanner ?: return
            val settings = ScanSettings.Builder()
                .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                .build()

            isScanning = true
            scanner.startScan(null, settings, object : ScanCallback() {
                override fun onScanResult(callbackType: Int, result: ScanResult?) {
                    val device = result?.device ?: return
                    val name = (device.name ?: result.scanRecord?.deviceName ?: "").uppercase()

                    if (name.contains("WHOOP")) {
                        try { scanner.stopScan(this) } catch (e: Exception) {}
                        isScanning = false

                        bluetoothGatt = device.connectGatt(
                            this@TelemetryService,
                            false,
                            gattCallback,
                            BluetoothDevice.TRANSPORT_LE
                        )
                    }
                }

                override fun onScanFailed(errorCode: Int) {
                    isScanning = false
                }
            })
        } catch (e: Exception) {
            isScanning = false
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                isWhoopConnected = true
                gatt?.discoverServices()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                isWhoopConnected = false
                liveHeartRate = 0
                gatt?.close()
                bluetoothGatt = null
            }
        }

        @SuppressLint("MissingPermission")
        override fun onServicesDiscovered(gatt: BluetoothGatt?, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                val service = gatt?.getService(HR_SERVICE_UUID)
                val characteristic = service?.getCharacteristic(HR_MEASUREMENT_UUID)

                if (characteristic != null) {
                    gatt.setCharacteristicNotification(characteristic, true)
                    val descriptor = characteristic.getDescriptor(CLIENT_CONFIG_DESCRIPTOR)
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                    } else {
                        @Suppress("DEPRECATION")
                        descriptor?.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                        @Suppress("DEPRECATION")
                        gatt.writeDescriptor(descriptor)
                    }
                }
            }
        }

        @Deprecated("Deprecated in API 33")
        override fun onCharacteristicChanged(gatt: BluetoothGatt?, characteristic: BluetoothGattCharacteristic?) {
            characteristic?.let { parseHeartRate(it.value) }
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            parseHeartRate(value)
        }
    }

    private fun parseHeartRate(value: ByteArray?) {
        if (value != null && value.isNotEmpty()) {
            val flags = value[0].toInt()
            val hr = if ((flags and 0x01) == 0) {
                value[1].toInt() and 0xFF
            } else {
                ((value[2].toInt() and 0xFF) shl 8) or (value[1].toInt() and 0xFF)
            }
            if (hr in 35..230) {
                liveHeartRate = hr
                isWhoopConnected = true
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun ensureObdConnected() {
        if (obdManager.isConnected) return

        try {
            val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
            val adapter = bluetoothManager.adapter
            if (adapter != null && adapter.isEnabled) {
                val pairedDevices = adapter.bondedDevices
                val obdDevice = pairedDevices.firstOrNull { device ->
                    val name = device.name?.uppercase() ?: ""
                    name.contains("OBD") || name.contains("ELM327") ||
                    name.contains("V-LINK") || name.contains("VGATE") || 
                    name.contains("OBDLINK") || name.contains("CAR_SCANNER")
                }
                if (obdDevice != null) {
                    obdManager.connect(obdDevice)
                }
            }
        } catch (e: Exception) {}
    }

    private fun startForegroundServiceNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "RoadSense Background Telemetry",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("RoadSense Active")
            .setContentText("Streaming real-time telemetry...")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    @SuppressLint("WakelockTimeout")
    private fun acquireWakeLock() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "RoadSense::BackgroundTelemetryLock")
        wakeLock?.acquire()
    }

    private var pollTick = 0

    private fun startStreamingLoop() {
        handler.post(object : Runnable {
            override fun run() {
                if (!isStreaming) return

                pollTick++
                if (pollTick % 6 == 0 && !isWhoopConnected) {
                    ensureWhoopConnected()
                }

                val currentHr = if (isWhoopConnected) liveHeartRate else 0

                val payload = JSONObject().apply {
                    put("lat", currentLat)
                    put("lon", currentLon)
                    put("gps_latitude", currentLat)
                    put("gps_longitude", currentLon)
                    put("accel_mag", accelMag)
                    put("gyro_mag", gyroMag)
                    put("obd_connected", obdManager.isConnected)
                    put("speed", if (obdManager.isConnected) obdManager.speedKmh else 0.0)
                    put("vehicle_speed", if (obdManager.isConnected) obdManager.speedKmh else 0.0)
                    put("rpm", if (obdManager.isConnected) obdManager.rpm else 0.0)
                    put("engine_speed", if (obdManager.isConnected) obdManager.rpm else 0.0)
                    put("throttle", if (obdManager.isConnected) obdManager.throttle else 0.0)
                    put("load", if (obdManager.isConnected) obdManager.engineLoad else 0.0)
                    put("heart_rate", currentHr)
                    put("bpm", currentHr)
                    put("hr", currentHr)
                }

                sendTelemetryToLaptop(payload.toString())
                handler.postDelayed(this, 300)
            }
        })
    }

    private fun sendTelemetryToLaptop(jsonString: String) {
        Thread {
            try {
                val mediaType = "application/json; charset=utf-8".toMediaType()
                val body = jsonString.toRequestBody(mediaType)
                val request = Request.Builder()
                    .url(targetNgrokUrl)
                    .post(body)
                    .build()

                httpClient.newCall(request).execute().close()
            } catch (e: Exception) {}
        }.start()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            if (it.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                accelMag = sqrt(it.values[0] * it.values[0] + it.values[1] * it.values[1] + it.values[2] * it.values[2]).toDouble()
            } else if (it.sensor.type == Sensor.TYPE_GYROSCOPE) {
                gyroMag = sqrt(it.values[0] * it.values[0] + it.values[1] * it.values[1] + it.values[2] * it.values[2]).toDouble()
            }
        }
    }

    override fun onLocationChanged(location: Location) {
        currentLat = location.latitude
        currentLon = location.longitude
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("MissingPermission")
    override fun onDestroy() {
        super.onDestroy()
        isStreaming = false
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
        obdManager.disconnect()
        bluetoothGatt?.close()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }
}