package io.github.masaponto.cleanphotos

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import android.widget.Toast
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Classifier(activity: Activity) {

    private val IMAGE_SIZE = 28
    //private val MODEL_NAME = "cnn.tflite"
    private val MODEL_NAME = "cnn_batch.tflite"
    private var tffile: Interpreter
    private var labelProbArray: Array<FloatArray>
    private var imageData: ByteBuffer

    private var batchLabelProbArray: Array<FloatArray>
    private var batchImageData: ByteBuffer

    init {
        tffile = Interpreter(loadModelFile(activity))
        labelProbArray = Array(1, {FloatArray(3)})
        imageData = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3 * 4) // 2352 * 4byte

        // for batch
        batchLabelProbArray = Array(3, {FloatArray(3)})
        batchImageData = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3 * 4 * 3) // 2352 * 4byte
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
        return labelProbArray
    }

    fun classifyBatchImage(matImages: Array<Mat?>) : Array<FloatArray> {

        var preds: Array<FloatArray?> = arrayOfNulls(matImages.size)

        batchImageData.order(ByteOrder.LITTLE_ENDIAN)
        for (matImage in matImages) {
            for (i in 0 until IMAGE_SIZE) {
                for (j in 0 until IMAGE_SIZE) {
                    for (k in 0 until 3) {
                        batchImageData.putFloat(matImage!!.get(i, j)[k].toFloat() / 255.0f)
                    }
                }
            }

        }

        tffile.run(batchImageData, batchLabelProbArray)
        batchImageData.clear()

        return batchLabelProbArray
    }

    fun classifyBitmapImage(bm: Bitmap): Int{
        val srcMat = Mat(bm.width, bm.height, CvType.CV_8UC3)
        Utils.bitmapToMat(bm, srcMat)
        Imgproc.cvtColor(srcMat,srcMat,Imgproc.COLOR_BGRA2RGB)

        // convert bitmap to Mat
        val matImage = Mat(28,28, CvType.CV_8UC3)
        Imgproc.resize(srcMat, matImage, Size(28.0, 28.0))
        matImage.convertTo(matImage, CvType.CV_8UC3)

        // classification with TF Lite
        val pred = classifyImage(matImage)

        // Release
        matImage.release()

        return toLabel(pred[0])
    }


    fun classifyImageFromPath(path: String): Int {
        val file = File(path)
        //val file = File(dir.absolutePath + "/test.jpg")

        if (!file.exists()) {
            throw Exception("Fail to load image")
        }

        // load image
        val bm = BitmapFactory.decodeFile(file.path)

        // convert bitmap to Mat
        val matImage = convertBitmap2Mat(bm)

        // classification with TF Lite
        val pred = classifyImage(matImage)

        // Release
        matImage.release()

        return toLabel(pred[0])
    }

    fun classifyBatchImageFromPath(paths: Array<String>): Array<Int> {
        val files = Array(paths.size) {File(paths[it])}
        var matImages: Array<Mat?> = arrayOfNulls(paths.size)

        for((index, file) in files.withIndex()) {

            if (!file.exists()) {
                throw Exception("Fail to load image")
            }

            val bm = BitmapFactory.decodeFile(file.path)
            matImages[index] = convertBitmap2Mat(bm)
        }

        val preds = classifyBatchImage(matImages)

        matImages.map { it?.release() }

        return preds.map { toLabel(it) }.toTypedArray()
    }

    private fun convertBitmap2Mat(bm: Bitmap): Mat {
        val srcMat = Mat(bm.width, bm.height, CvType.CV_8UC3)
        Utils.bitmapToMat(bm, srcMat)
        Imgproc.cvtColor(srcMat,srcMat,Imgproc.COLOR_BGRA2RGB)

        val matImage = Mat(28,28, CvType.CV_8UC3)
        Imgproc.resize(srcMat, matImage, Size(28.0, 28.0))
        matImage.convertTo(matImage, CvType.CV_8UC3)
        return matImage
    }

    private fun toLabel(floatArray: FloatArray): Int {
        val tmp = floatArray.indices.maxBy { floatArray[it] } ?: -1
        return tmp + 1
    }

}

