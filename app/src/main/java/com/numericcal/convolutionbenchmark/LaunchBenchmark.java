package com.numericcal.convolutionbenchmark;

import android.os.Handler;
import android.os.Message;

/**
 * Created by shaoyi on 1/20/2017.
 */

public class LaunchBenchmark implements Runnable {
    private int repeat;
    private int imgWidth;
    private int imgHeight;
    private int knlWidth;
    private int knlHeight;
    private int channel;
    private int numKnls;
    Handler handler;
    public LaunchBenchmark(int _repeat, int _imgWidth, int _imgHeight, int _knlWidth,
                           int _knlHeight, int _channel, int _numKnls, Handler _handler )
    {
        repeat = _repeat;
        imgWidth = _imgWidth;
        imgHeight = _imgHeight;
        knlWidth = _knlWidth;
        knlHeight = _knlHeight;
        channel = _channel;
        numKnls = _numKnls;
        handler = _handler;
    }
    @Override
    public void run() {
        String timeSpent = launchConvNumericcal(repeat,imgWidth,imgHeight,knlWidth,knlHeight,channel,numKnls);
        String benchmarkProperties = ""+ repeat+" Repetition--Input:";
        benchmarkProperties+=" "+imgHeight+"x"+imgWidth+"x"+channel+"; Kernel: "+numKnls+"x"+knlWidth+"x"+knlHeight+"x"+channel;

        String benchmarkReport = benchmarkProperties+"\nTime to completion: "+timeSpent;
        Message msg = Message.obtain();
        msg.obj = benchmarkReport;
        msg.setTarget(handler);

        msg.sendToTarget();


    }
    /*public native String launchConv(int repeat_, int imgWidth_, int imgHeight_,
                                    int knlWidth_, int knlHeight_, int channel_,
                                    int numKnls_);*/
    public native String launchConvNumericcal(int repeat_, int imgWidth_, int imgHeight_,
                                    int knlWidth_, int knlHeight_, int channel_,
                                    int numKnls_);


}
