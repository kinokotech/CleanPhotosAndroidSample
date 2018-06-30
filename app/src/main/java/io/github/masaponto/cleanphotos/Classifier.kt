package io.github.masaponto.cleanphotos

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

class Classifier(activity: Activity) {

    private val IMAGE_SIZE = 28
    private val MODEL_NAME = "cnn_bler.tflite"
    private var tffile: Interpreter
    private var labelProbArray: Array<FloatArray>
    private var imageData: ByteBuffer

    init {
        tffile = Interpreter(loadModelFile(activity))
        labelProbArray = Array(1, {FloatArray(1)})
        imageData = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3 * 4) // 2352 * 4byte
    }

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(MODEL_NAME)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun convertMatToTfliteInput(matImage: Mat) {
        imageData.order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until IMAGE_SIZE) {
            for (j in 0 until IMAGE_SIZE) {
                for (k in 0 until 3) {
                    imageData.putFloat(matImage.get(i, j)[k].toFloat() / 255.0f)
                }
            }
        }
    }

    fun classifyImage(matImage: Mat): Array<FloatArray> {
        convertMatToTfliteInput(matImage)
        tffile.run(imageData, labelProbArray)
        imageData.clear()
        Log.d("size",labelProbArray.size.toString())
        return labelProbArray
    }

}

