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
                val path = dir.absolutePath + "/IMG_20180425_122602.jpg"
                val label: Float? = classifier.classifyImageFromPath(path)

                Toast.makeText(this,
                        "Result:" + label.toString(),
                        Toast.LENGTH_LONG).show()
            }
        }
    }

}