package io.github.masaponto.cleanphotos

import java.nio.channels.FileChannel.MapMode.READ_ONLY
import android.content.res.AssetFileDescriptor
import android.app.Activity
import android.util.Log
import org.opencv.core.Mat
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Classifier(val activity: Activity) {

    val IMAGE_SIZE = 28
    private lateinit var tffile: Interpreter
    private var labelProbArray = Array(1, {FloatArray(1)})
    private val modelName = "cnn_bler.tflite"

    init {
        tffile = Interpreter(loadModelFile(activity))
    }

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun convertMattoTfliteInput(matImage: Mat) : ByteBuffer {
        var imageData = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3 * 4)
        imageData.order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until IMAGE_SIZE) {
            for (j in 0 until IMAGE_SIZE) {
                imageData.putFloat(matImage.get(i,j)[0].toFloat()/255.0f)
            }
        }
        return imageData
    }

    fun classifyImage(matImage: Mat): Array<FloatArray> {
        val imageData = convertMattoTfliteInput(matImage)
        tffile.run(imageData, labelProbArray)

        return labelProbArray
    }

}

