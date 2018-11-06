package io.github.masaponto.cleanphotos

import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.Toast
import org.opencv.android.OpenCVLoader
import java.io.File

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
                //val f = "IMG_20180825_095946.jpg" // dark
                //val f = "IMG_20180425_122602.jpg"  // normal
                val f = "test.jpg" // blur

                //resources.assets.open()
                val files = arrayOf("IMG_20180825_095946.jpg", "IMG_20180425_122602.jpg", "test.jpg")
                val paths = files.map { dir.absolutePath + "/" + it }.toTypedArray()

                //val path = dir.absolutePath + "/" + f
                //val label = classifier.classifyImageFromPath(path)

                val labels = classifier.classifyBatchImageFromPath(paths)

                
                Toast.makeText(this,
                        "Result0:" + labels[0] + "\n" +
                                "Result1:" + labels[1] + "\n" +
                                "Result2:" + labels[2] + "\n",
                        Toast.LENGTH_LONG).show()
            }
        }
    }

}