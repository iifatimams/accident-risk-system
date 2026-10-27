package ae.aus.roadsense

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.util.Log
import java.io.InputStream
import java.io.OutputStream
import java.util.UUID

class ObdManager {
    private val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    private var socket: BluetoothSocket? = null
    private var inputStream: InputStream? = null
    private var outputStream: OutputStream? = null

    @Volatile var speedKmh: Double = 0.0
    @Volatile var rpm: Double = 0.0
    @Volatile var throttle: Double = 0.0
    @Volatile var engineLoad: Double = 0.0
    @Volatile var isConnected: Boolean = false

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice): Boolean {
        return try {
            socket = device.createRfcommSocketToServiceRecord(SPP_UUID)
            socket?.connect()
            inputStream = socket?.inputStream
            outputStream = socket?.outputStream

            // Verify ELM327 response handshake
            val echoResp = sendObdCommand("ATE0")
            val resetResp = sendObdCommand("AT Z")

            if (echoResp.contains("OK") || resetResp.contains("ELM") || resetResp.contains("OK") || resetResp.contains("v")) {
                sendObdCommand("ATL0")    // Turn off linefeeds
                sendObdCommand("AT SP 0") // Auto-select vehicle OBD protocol
                isConnected = true
                Log.d("ObdManager", "ELM327 OBD hardware validated successfully.")
                true
            } else {
                Log.e("ObdManager", "Connected device is not an ELM327 OBD adapter.")
                disconnect()
                false
            }
        } catch (e: Exception) {
            Log.e("ObdManager", "OBD connection failed: ${e.message}")
            disconnect()
            false
        }
    }

    fun pollObdData() {
        if (!isConnected) return

        val speedResp = sendObdCommand("010D")
        speedKmh = parseSpeed(speedResp)

        val rpmResp = sendObdCommand("010C")
        rpm = parseRpm(rpmResp)

        val throttleResp = sendObdCommand("0111")
        throttle = parseSingleBytePercent(throttleResp, "4111")

        val loadResp = sendObdCommand("0104")
        engineLoad = parseSingleBytePercent(loadResp, "4104")
    }

    private fun sendObdCommand(command: String): String {
        return try {
            outputStream?.write((command + "\r").toByteArray())
            outputStream?.flush()

            val sb = StringBuilder()
            val buffer = ByteArray(256)
            val startTime = System.currentTimeMillis()

            while (System.currentTimeMillis() - startTime < 400) {
                if (inputStream?.available() ?: 0 > 0) {
                    val bytesRead = inputStream?.read(buffer) ?: 0
                    if (bytesRead > 0) {
                        val part = String(buffer, 0, bytesRead)
                        sb.append(part)
                        if (sb.contains(">")) break
                    }
                } else {
                    Thread.sleep(10)
                }
            }

            sb.toString()
                .replace(" ", "")
                .replace("\r", "")
                .replace("\n", "")
                .replace(">", "")
                .uppercase()
        } catch (e: Exception) {
            ""
        }
    }

    private fun parseSpeed(raw: String): Double {
        return try {
            val idx = raw.indexOf("410D")
            if (idx != -1 && raw.length >= idx + 6) {
                val hexValue = raw.substring(idx + 4, idx + 6)
                hexValue.toInt(16).toDouble()
            } else {
                speedKmh
            }
        } catch (e: Exception) {
            speedKmh
        }
    }

    private fun parseRpm(raw: String): Double {
        return try {
            val idx = raw.indexOf("410C")
            if (idx != -1 && raw.length >= idx + 8) {
                val hexValue = raw.substring(idx + 4, idx + 8)
                val a = hexValue.substring(0, 2).toInt(16)
                val b = hexValue.substring(2, 4).toInt(16)
                ((a * 256.0) + b) / 4.0
            } else {
                rpm
            }
        } catch (e: Exception) {
            rpm
        }
    }

    private fun parseSingleBytePercent(raw: String, header: String): Double {
        return try {
            val idx = raw.indexOf(header)
            if (idx != -1 && raw.length >= idx + header.length + 2) {
                val hexValue = raw.substring(idx + header.length, idx + header.length + 2)
                (hexValue.toInt(16) * 100.0) / 255.0
            } else {
                0.0
            }
        } catch (e: Exception) {
            0.0
        }
    }

    fun disconnect() {
        try {
            socket?.close()
        } catch (e: Exception) {}
        isConnected = false
        speedKmh = 0.0
        rpm = 0.0
        throttle = 0.0
        engineLoad = 0.0
    }
}