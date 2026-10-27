package ae.aus.roadsense

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
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
import android.os.ParcelUuid
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

    // WHOOP GATT Heart Rate Management
    @Volatile private var liveHeartRate: Int = 0
    private var isWhoopConnected = false
    private var bluetoothGatt: BluetoothGatt? = null
    private var bluetoothAdapter: BluetoothAdapter? = null

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
        connectToObdBluetooth()
        connectToWhoopGatt()
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
                Log.d("TelemetryService", "Initial location acquired: $currentLat, $currentLon")
            }

            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, this)
            locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 1000L, 0f, this)
        } catch (e: Exception) {
            Log.e("TelemetryService", "Location registration error: ${e.message}")
        }
    }

    @SuppressLint("MissingPermission")
    private fun connectToWhoopGatt() {
        try {
            val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
            bluetoothAdapter = bluetoothManager.adapter ?: return
            val scanner = bluetoothAdapter?.bluetoothLeScanner ?: return

            val filter = ScanFilter.Builder()
                .setServiceUuid(ParcelUuid(HR_SERVICE_UUID))
                .build()
            val settings = ScanSettings.Builder()
                .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                .build()

            scanner.startScan(listOf(filter), settings, object : ScanCallback() {
                override fun onScanResult(callbackType: Int, result: ScanResult?) {
                    val device = result?.device ?: return
                    val name = device.name?.uppercase() ?: ""

                    if (name.contains("WHOOP")) {
                        scanner.stopScan(this)
                        Log.d("TelemetryService", "Found WHOOP BLE device: ${device.name}. Connecting GATT...")
                        bluetoothGatt = device.connectGatt(this@TelemetryService, false, gattCallback)
                    }
                }
            })
        } catch (e: Exception) {
            Log.e("TelemetryService", "WHOOP GATT scan error: ${e.message}")
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                Log.d("TelemetryService", "WHOOP GATT Connected. Discovering services...")
                gatt?.discoverServices()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                isWhoopConnected = false
                liveHeartRate = 0
                Log.d("TelemetryService", "WHOOP GATT Disconnected.")
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
                    Log.d("TelemetryService", "Subscribed to WHOOP Heart Rate GATT notifications.")
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
            if (hr in 40..220) {
                liveHeartRate = hr
                isWhoopConnected = true
            }
        }
    }

    private fun getGenuineHeartRate(): Int {
        // Returns live WHOOP BLE pulse; returns 0 if disconnected
        return if (isWhoopConnected) liveHeartRate else 0
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
            .setContentText("Streaming real-time telemetry to inference engine...")
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

    @SuppressLint("MissingPermission")
    private fun connectToObdBluetooth() {
        Thread {
            try {
                val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
                bluetoothAdapter = bluetoothManager.adapter

                if (bluetoothAdapter != null && bluetoothAdapter!!.isEnabled) {
                    val pairedDevices = bluetoothAdapter!!.bondedDevices

                    val obdDevice = pairedDevices.firstOrNull { device ->
                        val name = device.name?.uppercase() ?: ""
                        name.contains("OBD") || name.contains("ELM327") ||
                        name.contains("V-LINK") || name.contains("VGATE") || name.contains("OBDLINK")
                    }

                    if (obdDevice != null) {
                        Log.d("TelemetryService", "Found target OBD hardware: ${obdDevice.name}. Connecting...")
                        obdManager.connect(obdDevice)
                    } else {
                        Log.d("TelemetryService", "No OBD hardware adapter found in paired devices.")
                    }
                }
            } catch (e: Exception) {
                Log.e("TelemetryService", "Bluetooth OBD connection failed: ${e.message}")
            }
        }.start()
    }

    private fun startStreamingLoop() {
        isStreaming = true
        handler.post(object : Runnable {
            override fun run() {
                if (!isStreaming) return

                if (obdManager.isConnected) {
                    Thread { obdManager.pollObdData() }.start()
                }

                val currentHr = getGenuineHeartRate()

                val payload = JSONObject().apply {
                    put("lat", currentLat)
                    put("lon", currentLon)
                    put("gps_latitude", currentLat)
                    put("gps_longitude", currentLon)
                    put("accel_mag", accelMag)
                    put("gyro_mag", gyroMag)
                    put("obd_connected", obdManager.isConnected)
                    put("speed", obdManager.speedKmh)
                    put("vehicle_speed", obdManager.speedKmh)
                    put("rpm", obdManager.rpm)
                    put("engine_speed", obdManager.rpm)
                    put("throttle", obdManager.throttle)
                    put("load", obdManager.engineLoad)
                    put("heart_rate", currentHr)
                    put("bpm", currentHr)
                    put("hr", currentHr)
                }

                sendTelemetryToLaptop(payload.toString())
                handler.postDelayed(this, 500)
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
            } catch (e: Exception) {
                // Ignore transient network errors
            }
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

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

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