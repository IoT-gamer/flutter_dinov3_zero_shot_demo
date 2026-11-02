package com.example.flutter_dinov3_zero_shot_demo

import android.content.Context
import androidx.annotation.Keep
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags

@Keep // Prevents ProGuard from stripping this class in release builds
class LitertBridge(applicationContext: Any, modelPath: String) {
    private var engine: Engine
    private var conversation: Conversation

    init {
        val context = applicationContext as Context
        val nativeLibDir = context.applicationInfo.nativeLibraryDir

        // Enable MTP via speculative decoding
        @OptIn(ExperimentalApi::class)
        ExperimentalFlags.enableSpeculativeDecoding = true

        // Configure the Engine
        val config = EngineConfig(
            modelPath = modelPath,
            // backend = Backend.NPU(nativeLibraryDir = nativeLibDir),
            backend = Backend.GPU(),
            visionBackend = Backend.GPU(),
            audioBackend = Backend.CPU()
        )

        // Initialize Engine and default Conversation
        engine = Engine(config)
        engine.initialize()
        conversation = engine.createConversation()
    }

    /**
     * Zero-shot object grounding using a single camera frame and either a recorded
     * 16kHz mono WAV audio file or a text target description.
     * Creates a short-lived Conversation so KV cache does not accumulate across enrollments.
     */
    fun groundObject(
        imagePath: String,
        audioPath: String,
        textPrompt: String
    ): String {
        val tempConversation = engine.createConversation()
        return try {
            val instruction = """
                Locate the target object requested by the user in this image.
                Respond ONLY with a JSON array of 4 integers [ymin, xmin, ymax, xmax] scaled from 0 to 1000.
                Example: [240, 310, 680, 720]
            """.trimIndent()

            val contents = if (audioPath.isNotEmpty()) {
                Contents.of(
                    Content.ImageFile(imagePath),
                    Content.AudioFile(audioPath),
                    Content.Text(instruction)
                )
            } else {
                Contents.of(
                    Content.ImageFile(imagePath),
                    Content.Text("Target object: $textPrompt\n$instruction")
                )
            }

            val responseMessage = tempConversation.sendMessage(contents)
            responseMessage.toString()
        } finally {
            tempConversation.close()
        }
    }

    // Accepts the raw image path directly from Flutter
    fun runVisionInference(prompt: String, imagePath: String): String {
        val imageContent = Content.ImageFile(imagePath)
        val textContent = Content.Text(prompt)
        val responseMessage = conversation.sendMessage(Contents.of(imageContent, textContent))
        return responseMessage.toString()
    }

    // Accepts the raw audio path directly from Flutter
    fun runAudioInference(prompt: String, audioPath: String): String {
        val audioContent = Content.AudioFile(audioPath)
        val textContent = Content.Text(prompt)
        val responseMessage = conversation.sendMessage(Contents.of(audioContent, textContent))
        return responseMessage.toString()
    }

    fun close() {
        conversation.close()
        engine.close()
    }
}