package io.github.masaponto.cleanphotos

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.os.ParcelFileDescriptor
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
                    Toast.makeText(this, "File found",Toast.LENGTH_LONG).show()

                    var option = BitmapFactory.Options()
                    option.inSampleSize = 2
                    option.outHeight = 28
                    option.outWidth = 28
                    //option.inJustDecodeBounds = false
                    //BitmapFactory.decodeFile(file.path, option)

                    //option.inJustDecodeBounds = true
                    //option.inSampleSize = 2

                    var bm = BitmapFactory.decodeFile(file.path, option)
                    bm = Bitmap.createScaledBitmap(bm, 28, 28, false)

                    Log.d("path", file.path)

                    if (bm != null) {
                        var matImage = Mat(28,28, CvType.CV_32F)
                        Utils.bitmapToMat(bm, matImage)

                        //matImage.reshape(28,28)

                        val label = classifier.classifyImage(matImage)

                        Log.d("Result", label.get(0).get(0).toString())


                        matImage.release()
                    }

                    //val imageView: ImageView = findViewById(R.id.imageView)
                    //imageView.setImageBitmap(bm)
                }

            }
        }
    }

    /*fun getBitmapFromUri(uri: Uri): Bitmap {
        val parcelfileDescriptor = contentResolver.openFileDescriptor(uri, "i")
        val fileDescriptor = parcelfileDescriptor.fileDescriptor
        val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)
        parcelfileDescriptor.close()
        return image
    }
*/
}