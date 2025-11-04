package com.example.chirpapp

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.media.*
import android.os.*
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : ComponentActivity() {

    private val sampleRate = 44100
    private val chirpStartHz = 18000.0
    private val chirpEndHz = 20000.0
    private val chirpDurationSec = 2.0

    private val status = mutableStateOf("Request Permission")
    private var recordJob: Job? = null
    private var playJob: Job? = null
    private val recordingBuffer = ByteArrayOutputStream()

    // Permission launchers
    private val requestMicPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                status.value = "Start Chirping"
                Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
            } else {
                status.value = "Permission Denied"
                Toast.makeText(this, "Microphone permission is required", Toast.LENGTH_LONG).show()
            }
        }

    private val requestAudioMediaPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (!isGranted) {
                Toast.makeText(this, "Storage permission required to save recordings", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
                requestAudioMediaPermission.launch(Manifest.permission.READ_MEDIA_AUDIO)
            }
        }

        setContent {
            val currentStatus = remember { status }
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Button(
                    onClick = {
                        when (currentStatus.value) {
                            "Start Chirping" -> {
                                startContinuousChirpAndRecord()
                                currentStatus.value = "Stop Chirping"
                            }
                            "Stop Chirping" -> {
                                stopChirpAndRecord()
                                currentStatus.value = "Start Chirping"
                            }
                            else -> {
                                requestMicPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            }
                        }
                    }
                ) {
                    Text(currentStatus.value)
                }
            }
        }

        if (hasRecordPermission()) {
            status.value = "Start Chirping"
        }
    }

    private fun hasRecordPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    // Start both chirping and recording
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startContinuousChirpAndRecord() {
        if (recordJob?.isActive == true) return
        recordingBuffer.reset()
        status.value = "Chirping & Recording..."

        // 🎙 Continuous recording coroutine
        recordJob = lifecycleScope.launch(Dispatchers.IO) {
            val minBuf = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBuf
            )
            val buffer = ByteArray(minBuf)
            recorder.startRecording()
            Log.d("ChirpApp", "Recording started")

            while (isActive) {
                val read = recorder.read(buffer, 0, buffer.size)
                if (read > 0) {
                    synchronized(recordingBuffer) {
                        recordingBuffer.write(buffer, 0, read)
                    }
                }
            }

            recorder.stop()
            recorder.release()
            Log.d("ChirpApp", "Recording stopped")

            val recordedBytes = synchronized(recordingBuffer) { recordingBuffer.toByteArray() }
            saveAsWav(recordedBytes)
        }

        // 🔊 Continuous chirp coroutine
        playJob = lifecycleScope.launch(Dispatchers.Default) {
            val chirpData = generateChirpBuffer()
            val track = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(chirpData.size * 2)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            track.play()
            Log.d("ChirpApp", "Continuous chirp started")

            // Repeatedly loop chirp buffer while active
            val shortBuf = chirpData
            while (isActive) {
                track.write(shortBuf, 0, shortBuf.size)
            }

            track.stop()
            track.release()
            Log.d("ChirpApp", "Chirp playback stopped")
        }
    }

    private fun stopChirpAndRecord() {
        playJob?.cancel()
        recordJob?.cancel()
        status.value = "Start Chirping"
        Log.d("ChirpApp", "Stopped chirping & recording")
    }

    // Generate one full chirp sweep buffer
    private fun generateChirpBuffer(): ShortArray {
        val totalSamples = (chirpDurationSec * sampleRate).toInt()
        val buffer = ShortArray(totalSamples)
        for (i in 0 until totalSamples) {
            val t = i.toDouble() / sampleRate
            val f = chirpStartHz + (chirpEndHz - chirpStartHz) * (t / chirpDurationSec)
            buffer[i] = (sin(2.0 * PI * f * t) * 32767.0).toInt().toShort()
        }
        return buffer
    }

    private fun saveAsWav(data: ByteArray) {
        val fileName = "chirp_recording_${System.currentTimeMillis()}.pmc"
        val channels = 1
        val bitDepth = 16
        val byteRate = sampleRate * channels * bitDepth / 8
        val totalDataLen = data.size.toLong()
        val totalWavLen = totalDataLen + 36

        val header = byteArrayOf(
            'R'.code.toByte(), 'I'.code.toByte(), 'F'.code.toByte(), 'F'.code.toByte(),
            (totalWavLen and 0xff).toByte(),
            (totalWavLen shr 8 and 0xff).toByte(),
            (totalWavLen shr 16 and 0xff).toByte(),
            (totalWavLen shr 24 and 0xff).toByte(),
            'W'.code.toByte(), 'A'.code.toByte(), 'V'.code.toByte(), 'E'.code.toByte(),
            'f'.code.toByte(), 'm'.code.toByte(), 't'.code.toByte(), ' '.code.toByte(),
            16, 0, 0, 0, 1, 0,
            channels.toByte(), 0,
            (sampleRate and 0xff).toByte(),
            (sampleRate shr 8 and 0xff).toByte(),
            (sampleRate shr 16 and 0xff).toByte(),
            (sampleRate shr 24 and 0xff).toByte(),
            (byteRate and 0xff).toByte(),
            (byteRate shr 8 and 0xff).toByte(),
            (byteRate shr 16 and 0xff).toByte(),
            (byteRate shr 24 and 0xff).toByte(),
            (channels * bitDepth / 8).toByte(), 0,
            bitDepth.toByte(), 0,
            'd'.code.toByte(), 'a'.code.toByte(), 't'.code.toByte(), 'a'.code.toByte(),
            (totalDataLen and 0xff).toByte(),
            (totalDataLen shr 8 and 0xff).toByte(),
            (totalDataLen shr 16 and 0xff).toByte(),
            (totalDataLen shr 24 and 0xff).toByte()
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val resolver = contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                put(MediaStore.Downloads.MIME_TYPE, "audio/pmc")
                put(MediaStore.Downloads.IS_PENDING, 1)
            }

            val downloads = MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val itemUri = resolver.insert(downloads, values)

            if (itemUri != null) {
                resolver.openOutputStream(itemUri)?.use { out ->
                    out.write(header)
                    out.write(data)
                }
                values.clear()
                values.put(MediaStore.Downloads.IS_PENDING, 0)
                resolver.update(itemUri, values, null, null)
                runOnUiThread {
                    Toast.makeText(this, "Saved to Downloads: $fileName", Toast.LENGTH_SHORT).show()
                }
                Log.i("ChirpApp", "Saved WAV to Downloads: $fileName")
            }
        } else {
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (!downloadsDir.exists()) downloadsDir.mkdirs()
            val file = File(downloadsDir, fileName)
            FileOutputStream(file).use { out ->
                out.write(header)
                out.write(data)
            }
            runOnUiThread {
                Toast.makeText(this, "Saved to: ${file.absolutePath}", Toast.LENGTH_SHORT).show()
            }
            Log.i("ChirpApp", "Saved PMC to Downloads: ${file.absolutePath}")
        }
    }
}
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