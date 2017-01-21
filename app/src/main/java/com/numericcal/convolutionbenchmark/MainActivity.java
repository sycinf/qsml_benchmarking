package com.numericcal.convolutionbenchmark;

import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    Spinner spinnerLib;
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
    static Handler handler;
    boolean benchmarkRunning;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        benchmarkRunning = false;
        // Example of a call to a native method
        //TextView tv = (TextView) findViewById(R.id.Result);
        //tv.setText(stringFromJNI());
        spinnerLib = (Spinner) findViewById(R.id.lib_selection_conv);
        ArrayAdapter<CharSequence> libAdapter = ArrayAdapter.createFromResource(this,
                R.array.conv_vendor_array, android.R.layout.simple_spinner_item);
        spinnerLib.setAdapter(libAdapter);
        handler = new Handler() {
            @Override
            public void handleMessage (Message msg) {
                TextView tv = (TextView) findViewById(R.id.Result);
                String original = tv.getText().toString();
                tv.setText(original+msg.obj.toString()+"\n");
                benchmarkRunning = false;
            }
        };
    }
    public void onRunConvClick(View view)
    {
        if (benchmarkRunning)
                return;
        else
            benchmarkRunning = true;

        EditText widthET = (EditText) findViewById(R.id.convWidth);
        EditText heightET = (EditText) findViewById(R.id.convHeight);
        EditText depthET = (EditText) findViewById(R.id.convDepth);

        EditText kWidthET = (EditText) findViewById(R.id.kernelWidth);
        EditText kHeightET = (EditText) findViewById(R.id.kernelHeight);
        EditText kNumberET = (EditText)findViewById(R.id.kernelNumber);

        EditText repeatET = (EditText)findViewById(R.id.repeat);

        String errorMsg = "Image/Kernel width, height, depth should all be positive integers";
        int imgWidth=0, imgHeight=0, imgDepth=0;
        int kWidth=0, kHeight=0, kNumber=0;
        int repetition = 0;

        Toast error = Toast.makeText(getApplicationContext(),errorMsg, Toast.LENGTH_SHORT);
        try {
            imgWidth = Integer.valueOf(widthET.getText().toString());
            imgHeight = Integer.valueOf(heightET.getText().toString());
            imgDepth = Integer.valueOf(depthET.getText().toString());
            kWidth = Integer.valueOf(kWidthET.getText().toString());
            kHeight = Integer.valueOf(kHeightET.getText().toString());
            kNumber = Integer.valueOf(kNumberET.getText().toString());
            repetition = Integer.valueOf(repeatET.getText().toString());
        }catch(Exception e)
        {
            //error.show();
            //TextView tv = (TextView) findViewById(R.id.Result);
            //tv.setText(benchmarkHardCoded());

        }
        String lib = spinnerLib.getSelectedItem().toString();

        if(imgWidth<=0 || imgHeight <= 0 || imgDepth <= 0 || kWidth <= 0 || kHeight<=0 ||
                kNumber <=0 || repetition <=0 )
        {
            //error.show();
        }

        Thread newThread = new Thread(new LaunchBenchmark(repetition,imgWidth,imgHeight,
                kWidth,kHeight,imgDepth,kNumber,handler));
        newThread.start();


    }


}
