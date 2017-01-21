#include <jni.h>
#include <string>
#include <sstream>


#define NANOS 1000000000LL
/* helper function for timing */
struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    return start_time;
}

long long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_REALTIME , &end_time);
    long long start, end;
    start = start_time.tv_sec*NANOS+start_time.tv_nsec;
    end = end_time.tv_sec*NANOS+end_time.tv_nsec;
    long long diffInNanos = end - start;
    return diffInNanos;
}

/* helper */
float* allocateAndPopulate(int num, int width, int height, int channel)
{
    float* array = new float[num*width*height*channel];
    for(int x=0; x<num; x++)
        for(int i=0; i<width; i++)
            for(int j=0; j<height; j++)
                for(int k=0; k<channel; k++)
                    array[i*height*channel+j*channel+k] = i*height*channel+j*channel+k;
    return array;
}
void deallocate(float* ptr)
{
    delete [] ptr;
}

extern "C"
{

    jstring
    Java_com_numericcal_convolutionbenchmark_LaunchBenchmark_launchConv(
            JNIEnv *env,
            jobject, /* this */
            int repeat, int imgWidth, int imgHeight,
            int knlWidth, int knlHeight, int channel,
            int numKnls)
    {
        std::stringstream rtStringStream;
        float* inputImage = allocateAndPopulate( 1, imgWidth, imgHeight,channel);
        float* kernels = allocateAndPopulate(numKnls, knlWidth,knlHeight,channel);
        int outputWidth = imgWidth-knlWidth+1;
        int outputHeight = imgHeight-knlHeight+1;
        float* output = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1);
        struct timespec timecount;
        timecount = timer_start();
        for(int repInd = 0; repInd < repeat; repInd++)
        {

        }
        long long timeSpent = timer_end(timecount);
        deallocate(inputImage);
        deallocate(kernels);
        deallocate(output);
        rtStringStream<<timeSpent;
        return env->NewStringUTF(rtStringStream.str().c_str());
    }
}
