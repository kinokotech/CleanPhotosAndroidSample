package io.github.masaponto.cleanphotos

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import java.io.File
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if(!OpenCVLoader.initDebug()){
            Log.i("OpenCV", "Failed")
        } else{
            Log.i("OpenCV", "successfully built !")
        }

        val button: Button = findViewById(R.id.button)
        val classifier = Classifier(this@MainActivity)
        val dir = File(Environment.getExternalStorageDirectory().path + "/DCIM/Camera")

        button.setOnClickListener {
            if (dir.exists()) {

                //val file = File(dir.absolutePath + "/IMG_20180425_122554.jpg")
                val file = File(dir.absolutePath + "/IMG_20180425_122602.jpg")
                //val file = File(dir.absolutePath + "/test.jpg")

                if (file.exists()) {

                    val bm = BitmapFactory.decodeFile(file.path)

                    bm?.let {

                        val srcMat = Mat(bm.width, bm.height, CvType.CV_8UC3)
                        Utils.bitmapToMat(it, srcMat)
                        Imgproc.cvtColor(srcMat,srcMat,Imgproc.COLOR_BGRA2RGB)

                        // convert bitmap to Mat
                        val matImage = Mat(28,28, CvType.CV_8UC3)
                        Imgproc.resize(srcMat, matImage, Size(28.0, 28.0))
                        matImage.convertTo(matImage, CvType.CV_8UC3)

                        // classification with TF Lite
                        val label = classifier.classifyImage(matImage)

                        // Release
                        matImage.release()

                        Toast.makeText(this,
                                "Result:" + label[0][0].toString(),
                                Toast.LENGTH_LONG).show()

                    }
                }

            }
        }
    }

}