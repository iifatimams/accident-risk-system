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

    private val lock = Any()
    private var lastValidSpeedTime = 0L
    private var lastValidRpmTime = 0L
    private val STALE_THRESHOLD_MS = 1500L

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice): Boolean {
        synchronized(lock) {
            disconnect()
            return try {
                val sock = device.createRfcommSocketToServiceRecord(SPP_UUID)
                sock.connect()
                socket = sock
                inputStream = sock.inputStream
                outputStream = sock.outputStream

                // Configuration handshake
                sendObdCommand("ATZ")
                Thread.sleep(150)
                sendObdCommand("ATE0") // Echo off
                sendObdCommand("ATL0") // Linefeeds off
                sendObdCommand("ATS0") // Spaces off
                sendObdCommand("ATSP0") // Auto protocol

                val probe = sendObdCommand("0100")
                if (probe.contains("4100") || probe.contains("41 00")) {
                    isConnected = true
                    val now = System.currentTimeMillis()
                    lastValidSpeedTime = now
                    lastValidRpmTime = now
                    Log.d("ObdManager", "ELM327 OBD ECU validated successfully.")
                    true
                } else {
                    Log.w("ObdManager", "ELM327 answered, but vehicle ECU did not: $probe")
                    disconnect()
                    false
                }
            } catch (e: Exception) {
                Log.e("ObdManager", "OBD connect failed: ${e.message}")
                disconnect()
                false
            }
        }
    }

    fun pollObdData() {
        synchronized(lock) {
            if (!isConnected || socket == null) return

            val now = System.currentTimeMillis()

            // 1. Poll Speed (010D)
            val speedResp = sendObdCommand("010D")
            val parsedSpeed = parseSpeed(speedResp)
            if (parsedSpeed != null) {
                speedKmh = parsedSpeed
                lastValidSpeedTime = now
            } else if (now - lastValidSpeedTime > STALE_THRESHOLD_MS) {
                speedKmh = 0.0
            }

            // 2. Poll RPM (010C)
            val rpmResp = sendObdCommand("010C")
            val parsedRpm = parseRpm(rpmResp)
            if (parsedRpm != null) {
                rpm = parsedRpm
                lastValidRpmTime = now
            } else if (now - lastValidRpmTime > STALE_THRESHOLD_MS) {
                rpm = 0.0
            }

            // 3. Poll Throttle (0111)
            val throttleResp = sendObdCommand("0111")
            val parsedThrottle = parseSingleBytePercent(throttleResp, "4111")
            if (parsedThrottle != null) {
                throttle = parsedThrottle
            }

            // 4. Poll Engine Load (0104)
            val loadResp = sendObdCommand("0104")
            val parsedLoad = parseSingleBytePercent(loadResp, "4104")
            if (parsedLoad != null) {
                engineLoad = parsedLoad
            }
        }
    }

    private fun sendObdCommand(command: String): String {
        val out = outputStream ?: return ""
        val input = inputStream ?: return ""

        return try {
            out.write((command + "\r").toByteArray())
            out.flush()

            val sb = StringBuilder()
            val buffer = ByteArray(256)
            val startTime = System.currentTimeMillis()

            while (System.currentTimeMillis() - startTime < 350) {
                if (input.available() > 0) {
                    val bytesRead = input.read(buffer)
                    if (bytesRead > 0) {
                        val part = String(buffer, 0, bytesRead)
                        sb.append(part)
                        if (sb.contains(">")) break
                    }
                } else {
                    Thread.sleep(8)
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

    private fun parseSpeed(raw: String): Double? {
        if (raw.isEmpty()) return null
        return try {
            val idx = raw.indexOf("410D")
            if (idx != -1 && raw.length >= idx + 6) {
                val hexValue = raw.substring(idx + 4, idx + 6)
                hexValue.toInt(16).toDouble()
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun parseRpm(raw: String): Double? {
        if (raw.isEmpty()) return null
        return try {
            val idx = raw.indexOf("410C")
            if (idx != -1 && raw.length >= idx + 8) {
                val hexValue = raw.substring(idx + 4, idx + 8)
                val a = hexValue.substring(0, 2).toInt(16)
                val b = hexValue.substring(2, 4).toInt(16)
                ((a * 256.0) + b) / 4.0
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun parseSingleBytePercent(raw: String, header: String): Double? {
        if (raw.isEmpty()) return null
        return try {
            val idx = raw.indexOf(header)
            if (idx != -1 && raw.length >= idx + header.length + 2) {
                val hexValue = raw.substring(idx + header.length, idx + header.length + 2)
                (hexValue.toInt(16) * 100.0) / 255.0
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    fun disconnect() {
        synchronized(lock) {
            try {
                socket?.close()
            } catch (e: Exception) {}
            socket = null
            inputStream = null
            outputStream = null
            isConnected = false
            speedKmh = 0.0
            rpm = 0.0
            throttle = 0.0
            engineLoad = 0.0
        }
    }
}