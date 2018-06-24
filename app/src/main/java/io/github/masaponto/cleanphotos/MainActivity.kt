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
                val file = File(dir.absolutePath + "/IMG_20180425_122602.jpg")
                //val file = File(dir.absolutePath + "/test.jpg")
                if (file.exists()) {

                    val option = BitmapFactory.Options()
                    option.inSampleSize = 2

                    //option.outHeight = 28
                    //option.outWidth = 28
                    //option.inJustDecodeBounds = false
                    //BitmapFactory.decodeFile(file.path, option)
                    //option.inJustDecodeBounds = true
                    //option.inSampleSize = 2

                    var bm = BitmapFactory.decodeFile(file.path, option)
                    bm = Bitmap.createScaledBitmap(bm, 28, 28, false)

                    bm?.let {

                        // set image view
                        val imageView: ImageView = findViewById(R.id.imageView)
                        imageView.setImageBitmap(bm)

                        // convert bitmap to Mat
                        val matImage = Mat(28,28, CvType.CV_32F)
                        Utils.bitmapToMat(it, matImage)

                        // classification with TF Lite
                        val label = classifier.classifyImage(matImage)

                        // Release
                        matImage.release()

                        Toast.makeText(this,
                                "Result:" + label[0][0].toString(),Toast.LENGTH_LONG).show()

                    }

                }

            }
        }
    }

}