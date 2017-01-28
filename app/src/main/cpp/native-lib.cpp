#define RUNDEVICE
#define MAXELE 20
//#define DEBUGNEON
#define USENEON
#define CHANNELUNROLLFACTOR 8

#define LOGCATDEBUG
#include <string>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#ifdef RUNDEVICE
#include <jni.h>
#include <qsml.h>
  #ifdef LOGCATDEBUG
    #include <android/log.h>

    #define  LOG_TAG    "MINEDEBUG:"

    #define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
    #define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
  #endif
  #ifdef USENEON
    #include <arm_neon.h>
  #endif
#endif


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
std::stringstream rtStringStream;
int made;

/* end of timing */
enum populateStyle
{
    FILLZERO,
    FILLRAND,
    FILLPOSITION
};

/* helper */
float* allocateAndPopulate(int num, int width, int height, int channel, populateStyle curPopStyle)
{
    float* array = new float[num*width*height*channel];
    for(int x=0; x<num; x++)
        for(int i=0; i<width; i++)
            for(int j=0; j<height; j++)
                for(int k=0; k<channel; k++) {
                    float fillNum = 0.0;
                    if(curPopStyle == FILLPOSITION)
                        fillNum = (x * width * height * channel + i * height * channel + j * channel + k)%MAXELE;
                    else if(curPopStyle == FILLRAND)
                        fillNum = (float) (rand() %  (MAXELE));

                    array[x * width * height * channel+ i * height * channel + j * channel + k] = fillNum;
                }
    return array;
}
void repopulate(float* array, int num, int width, int height, int channel, populateStyle curPopStyle)
{
    for(int x=0; x<num; x++)
        for(int i=0; i<width; i++)
            for(int j=0; j<height; j++)
                for(int k=0; k<channel; k++) {
                    float fillNum = 0.0;
                    if(curPopStyle == FILLPOSITION)
                        fillNum = (i * height * channel + j * channel + k)%MAXELE;
                    else if(curPopStyle == FILLRAND)
                        fillNum = (float) (rand() %  (MAXELE));

                    array[i * height * channel + j * channel + k] = fillNum;
                }
}
void deallocate(float* ptr)
{
    delete [] ptr;
}
// numericcal specific stuff
/*float channelReduction(float* imgChannel, float* curKernelEle, int channel )
{
    int numB = channel/16;
    float final =0.0;

    for(int i=0; i<numB; i+=16)
    {
        float* curI = imgChannel+i;
        float* curK = curKernelEle+i;

        float32x4_t input1 = vld1q_f32(curI);
        float32x4_t kernel1 = vld1q_f32(curK);

        float32x4_t input2 = vld1q_f32(curI+4);
        float32x4_t kernel2 = vld1q_f32(curK+4);

        float32x4_t input3 = vld1q_f32(curI+8);
        float32x4_t kernel3 = vld1q_f32(curK+8);


        float32x4_t input4 = vld1q_f32(curI+12);
        float32x4_t kernel4 = vld1q_f32(curK+12);

        float32x4_t result1 = vmulq_f32(kernel1,input1);
        float32x4_t result2 = vmlaq_f32(result1,kernel2,input2);
        float32x4_t result3 = vmlaq_f32(result2,kernel3,input3);
        float32x4_t result4 = vmlaq_f32(result3,kernel4,input4);

        float add2Final =vaddvq_f32(result4);
        final+=add2Final;
        // reduction
    }
    return final;
}

void topRowChannel(float* image, float* kernel, float* output, int imgDIM, int kDIM, int channel)
{
    float* curInPixel = image;
    float* kernel0thPixel = kernel;
    output[0] = channelReduction(curInPixel,kernel0thPixel,channel);

    float * kernel1thPixel = kernel+channel;
    curInPixel = curInPixel+channel;
    output[0]+= channelReduction(curInPixel,kernel1thPixel, channel);
    output[1]+= channelReduction(curInPixel,kernel0thPixel, channel);
    curInPixel = curInPixel+channel;
    float * kernel2thPixel = kernel+2*channel;

    int steadyStateNum = imgDIM-2*(kDIM-1);
    for(int i=0; i< steadyStateNum; i++) {
        output[i+0] += channelReduction(curInPixel, kernel2thPixel, channel);
        output[i+1] += channelReduction(curInPixel, kernel1thPixel, channel);
        output[i+2] += channelReduction(curInPixel, kernel0thPixel, channel);
        curInPixel = curInPixel+channel;
    }
    // end of the row
    output[steadyStateNum]+= channelReduction(curInPixel,kernel2thPixel, channel);
    output[steadyStateNum+1]+= channelReduction(curInPixel,kernel1thPixel, channel);
    curInPixel = curInPixel+channel;
    output[steadyStateNum+1]+= channelReduction(curInPixel,kernel2thPixel, channel);
}

void secondRowChannel(float* imageRow, float* kernel, float* output, int imgDIM, int kDIM, int channel)
{
    int outDIM = imgDIM-kDIM+1;
    float* curInPixel = imageRow;
    float* kernel0thPixel = kernel;
    float* kernel1thPixel = kernel+channel;
    float* kernel3thPixel = kernel+3*channel;
    float* kernel4thPixel = kernel+4*channel;
    output[0] += channelReduction(curInPixel,kernel3thPixel,channel);
    output[outDIM] += channelReduction(curInPixel,kernel0thPixel,channel);

    curInPixel += channel;
    output[0] += channelReduction(curInPixel,kernel4thPixel,channel);
    output[1] += channelReduction(curInPixel,kernel3thPixel,channel);
    output[outDIM] += channelReduction(curInPixel,kernel1thPixel,channel);
    output[outDIM+1] += channelReduction(curInPixel,kernel0thPixel,channel);

    float* kernel5thPixel = kernel+5*channel;
    float* kernel2thPixel = kernel+2*channel;
    int steadyStateNum = imgDIM-2*(kDIM-1);
    for(int i=0; i< steadyStateNum; i++) {
        output[i+0] += channelReduction(curInPixel, kernel5thPixel, channel);
        output[i+1] += channelReduction(curInPixel, kernel4thPixel, channel);
        output[i+2] += channelReduction(curInPixel, kernel3thPixel, channel);

        output[outDIM+i+0] += channelReduction(curInPixel, kernel2thPixel, channel);
        output[outDIM+i+1] += channelReduction(curInPixel, kernel1thPixel, channel);
        output[outDIM+i+2] += channelReduction(curInPixel, kernel0thPixel, channel);

        curInPixel = curInPixel+channel;
    }
    output[steadyStateNum]+= channelReduction(curInPixel,kernel5thPixel, channel);
    output[steadyStateNum+1]+= channelReduction(curInPixel,kernel4thPixel, channel);
    output[steadyStateNum+outDIM] += channelReduction(curInPixel,kernel2thPixel,channel);
    output[steadyStateNum+outDIM+1] += channelReduction(curInPixel,kernel1thPixel,channel);

    curInPixel = curInPixel+channel;
    output[steadyStateNum+1]+= channelReduction(curInPixel,kernel5thPixel, channel);
    output[steadyStateNum+outDIM+1]+= channelReduction(curInPixel,kernel2thPixel, channel);
}

void steadyOne2Nine(float* imageRow, float* kernel, float* output, int imgDIM, int kDIM, int channel)
{
    int outDIM = imgDIM-kDIM+1;
    float* curInPixel = imageRow;
    float* kernel0thPixel = kernel;
    float* kernel1thPixel = kernel+channel;
    float* kernel2thPixel = kernel+2*channel;
    float* kernel3thPixel = kernel+3*channel;
    float* kernel4thPixel = kernel+4*channel;
    float* kernel5thPixel = kernel+5*channel;
    float* kernel6thPixel = kernel+6*channel;
    float* kernel7thPixel = kernel+7*channel;
    float* kernel8thPixel = kernel+8*channel;

    output[0] += channelReduction(curInPixel,kernel6thPixel,channel);
    output[outDIM] += channelReduction(curInPixel,kernel3thPixel,channel);
    output[outDIM*2] += channelReduction(curInPixel,kernel0thPixel,channel);

    curInPixel = curInPixel+channel;
    output[0] += channelReduction(curInPixel,kernel7thPixel,channel);
    output[1] += channelReduction(curInPixel,kernel6thPixel,channel);
    output[outDIM] += channelReduction(curInPixel,kernel4thPixel,channel);
    output[outDIM+1] += channelReduction(curInPixel,kernel3thPixel,channel);
    output[outDIM*2] += channelReduction(curInPixel,kernel1thPixel,channel);
    output[outDIM*2+1] += channelReduction(curInPixel,kernel0thPixel,channel);

    curInPixel = curInPixel+channel;
    int steadyStateNum = imgDIM-2*(kDIM-1);
    for(int i=0; i< steadyStateNum; i++) {
        output[0+i] += channelReduction(curInPixel,kernel8thPixel,channel);
        output[1+i] += channelReduction(curInPixel,kernel7thPixel,channel);
        output[2+i] += channelReduction(curInPixel,kernel6thPixel,channel);
        output[outDIM+i] += channelReduction(curInPixel,kernel5thPixel,channel);
        output[outDIM+i+1] += channelReduction(curInPixel,kernel4thPixel,channel);
        output[outDIM+i+2] += channelReduction(curInPixel,kernel3thPixel,channel);

        output[outDIM*2+i] += channelReduction(curInPixel,kernel2thPixel,channel);
        output[outDIM*2+i+1] += channelReduction(curInPixel,kernel1thPixel,channel);
        output[outDIM*2+i+2] += channelReduction(curInPixel,kernel0thPixel,channel);
        curInPixel = curInPixel+channel;
    }
    output[steadyStateNum] += channelReduction(curInPixel,kernel8thPixel,channel);
    output[steadyStateNum+1] += channelReduction(curInPixel,kernel7thPixel,channel);
    output[steadyStateNum+outDIM] += channelReduction(curInPixel,kernel5thPixel,channel);
    output[steadyStateNum+outDIM+1] += channelReduction(curInPixel,kernel4thPixel,channel);
    output[steadyStateNum+outDIM*2] += channelReduction(curInPixel,kernel2thPixel,channel);
    output[steadyStateNum+outDIM*2+1] += channelReduction(curInPixel,kernel1thPixel,channel);
    curInPixel = curInPixel+channel;
    output[steadyStateNum+1] += channelReduction(curInPixel,kernel8thPixel,channel);
    output[steadyStateNum+outDIM+1] += channelReduction(curInPixel,kernel5thPixel,channel);
    output[steadyStateNum+outDIM*2+1] += channelReduction(curInPixel,kernel2thPixel,channel);

}

void channelMajorVector(float* image, float* kernel, float* output, int imgDIM, int kDIM, int channel)
{
    int outDim = imgDIM-kDIM+1;
    topRowChannel(image,kernel,output,imgDIM,kDIM,channel);
    secondRowChannel(image+channel*imgDIM, kernel, output, imgDIM, kDIM, channel);
    int steadyStateNum = imgDIM-2*(kDIM-1);
    float* curImagRow = image+channel*imgDIM*2;
    float* curOutputRow = output;
    for(int i=0; i<steadyStateNum; i++)
    {
        steadyOne2Nine(curImagRow, kernel, curOutputRow, imgDIM, kDIM, channel);
        curImagRow+=channel*imgDIM;
        curOutputRow+=outDim;
    }
    // last part...
}
*/

/*

void oneEleOneRow(float* inputRow, int numMul, float* outputRow, float element)
{
    int numQ = numMul/4;

    for(int i=0; i<numQ*4; i+=4)
    {
        float* cur = inputRow+i;
        float* out = outputRow+i;
        float32x4_t eleDup = vld1q_dup_f32(&element);
        float32x4_t input = vld1q_f32(cur);
        float32x4_t output =  vld1q_f32(out);
        float32x4_t result = vmlaq_f32(output,eleDup,input);
        vst1q_f32(out,result);
    }
    for(int i=numQ*4; i<numMul; i++)
    {
        outputRow[i]+=inputRow[i]*element;
    }
    //for(int i=0; i<numMul; i++)
    //{
    //    outputRow[i]+=inputRow[i]*element;
    //}
}

void oneEleOneRow1(float* inputRow, int numMul, float* outputRow, float element)
{
    int numQ = numMul/28;
    float32x4_t eleDup = vld1q_dup_f32(&element);
    __builtin_prefetch(inputRow);
    __builtin_prefetch(outputRow);
    for(int i=0; i<numQ*28; i+=28)
    {
        float* cur = inputRow+i;
        float* out1 = outputRow+i;

        float32x4_t input1 = vld1q_f32(cur);
        float32x4_t output1 =  vld1q_f32(out1);
        float* out2 = out1+4;
        float32x4_t input2 = vld1q_f32(cur+4);
        float32x4_t output2 =  vld1q_f32(out2);
        float* out3 = out1+8;
        float32x4_t input3 = vld1q_f32(cur+8);
        float32x4_t output3 =  vld1q_f32(out3);
        float* out4 = out1+12;
        float32x4_t input4 = vld1q_f32(cur+12);
        float32x4_t output4 =  vld1q_f32(out4);
        float* out5 = out1+16;
        float32x4_t input5 = vld1q_f32(cur+16);
        float32x4_t output5 =  vld1q_f32(out5);

        float* out6 = out1+20;
        float32x4_t input6 = vld1q_f32(cur+20);
        float32x4_t output6 =  vld1q_f32(out6);
        float* out7 = out1+28;
        float32x4_t input7 = vld1q_f32(cur+24);
        float32x4_t output7 =  vld1q_f32(out7);




        float32x4_t result1 = vmlaq_f32(output1,eleDup,input1);
        float32x4_t result2 = vmlaq_f32(output2,eleDup,input2);
        float32x4_t result3 = vmlaq_f32(output3,eleDup,input3);
        float32x4_t result4 = vmlaq_f32(output4,eleDup,input4);
        float32x4_t result5 = vmlaq_f32(output5,eleDup,input5);
        float32x4_t result6 = vmlaq_f32(output6,eleDup,input6);
        float32x4_t result7 = vmlaq_f32(output7,eleDup,input7);
        vst1q_f32(out1,result1);
        vst1q_f32(out2,result2);
        vst1q_f32(out3,result3);
        vst1q_f32(out4,result4);
        vst1q_f32(out5,result5);
        vst1q_f32(out6,result6);
        vst1q_f32(out7,result7);
    }
    int  doneI = numQ*28;
    oneEleOneRow(inputRow+doneI,numMul-doneI,outputRow+doneI,element);
    //for(int i=numQ*28; i<numMul; i++)
    //{
    //    oneEleOneRow(inputRow+i,numMul-i,outputRow+i,element);

    //}

}




void run3by3KernelByRow(float* image, float* kernel, float* output, int imgDIM, int kDIM)
{
    float* output1Row = output;
    float* input1Row = image;
    int rowMul = imgDIM-kDIM+1;

    int offsetIn = 0;
    int offsetOut = 0;

    oneEleOneRow1(input1Row,rowMul,output1Row,kernel[0]);
    oneEleOneRow1(input1Row+1,rowMul, output1Row,kernel[1]);
    oneEleOneRow1(input1Row+2, rowMul, output1Row, kernel[2]);

    oneEleOneRow1(input1Row + imgDIM, rowMul, output1Row, kernel[3]);
    oneEleOneRow1(input1Row + imgDIM+1, rowMul,output1Row, kernel[4]);
    oneEleOneRow1(input1Row + imgDIM+2, rowMul,output1Row, kernel[5]);

    oneEleOneRow1(input1Row + imgDIM, rowMul, output1Row+rowMul, kernel[0]);
    oneEleOneRow1(input1Row + imgDIM+1, rowMul,output1Row+rowMul, kernel[1]);
    oneEleOneRow1(input1Row + imgDIM+2, rowMul,output1Row+rowMul, kernel[2]);

    offsetIn+=imgDIM;
    int bound = imgDIM-2*(kDIM-1);
    //steady state
    for(int k=0; k< bound ; k++) {
        offsetIn += imgDIM;
        oneEleOneRow1(input1Row + offsetIn, rowMul, output1Row+offsetOut, kernel[6]);
        oneEleOneRow1(input1Row + offsetIn+1, rowMul,output1Row+offsetOut, kernel[7]);
        oneEleOneRow1(input1Row + offsetIn+2, rowMul,output1Row+offsetOut, kernel[8]);
        offsetOut += rowMul;
        oneEleOneRow1(input1Row + offsetIn, rowMul, output1Row+offsetOut, kernel[3]);
        oneEleOneRow1(input1Row + offsetIn+1, rowMul, output1Row+offsetOut, kernel[4]);
        oneEleOneRow1(input1Row + offsetIn+2, rowMul, output1Row+offsetOut, kernel[5]);
        //offsetIn += imgDIM;
        oneEleOneRow1(input1Row + offsetIn, rowMul, output1Row+offsetOut+rowMul, kernel[0]);
        oneEleOneRow1(input1Row + offsetIn+1, rowMul,output1Row+offsetOut+rowMul, kernel[1]);
        oneEleOneRow1(input1Row + offsetIn+2, rowMul,output1Row+offsetOut+rowMul, kernel[2]);

    }
    offsetIn += imgDIM;
    oneEleOneRow1(input1Row + offsetIn, rowMul, output1Row+offsetOut, kernel[6]);
    oneEleOneRow1(input1Row + offsetIn+1, rowMul,output1Row+offsetOut, kernel[7]);
    oneEleOneRow1(input1Row + offsetIn+2, rowMul,output1Row+offsetOut, kernel[8]);
    offsetOut+=rowMul;
    oneEleOneRow1(input1Row + offsetIn, rowMul, output1Row+offsetOut, kernel[3]);
    oneEleOneRow1(input1Row + offsetIn+1, rowMul, output1Row+offsetOut, kernel[4]);
    oneEleOneRow1(input1Row + offsetIn+2, rowMul, output1Row+offsetOut, kernel[5]);
    offsetIn += imgDIM;
    oneEleOneRow1(input1Row + offsetIn, rowMul, output1Row+offsetOut, kernel[6]);
    oneEleOneRow1(input1Row + offsetIn+1, rowMul,output1Row+offsetOut, kernel[7]);
    oneEleOneRow1(input1Row + offsetIn+2, rowMul,output1Row+offsetOut, kernel[8]);
}
*/

/* Numericcal -- channel major */
/*
void numericcalChannelMajor(float* inputImage, int imgWidth, int imgHeight, int channel,
                            float* kernels, int numKnls, int knlWidth, int knlHeight,
                            float* output, int outputWidth, int outputHeight)
{
    if(outputWidth!=imgWidth-knlWidth+1 || outputHeight!= imgHeight-knlHeight+1)
        return;
    // special treatment for top


    int outDim = imgDIM-kDIM+1;
    topRowChannel(image,kernel,output,imgDIM,kDIM,channel);
    secondRowChannel(image+channel*imgDIM, kernel, output, imgDIM, kDIM, channel);
    int steadyStateNum = imgDIM-2*(kDIM-1);
    float* curImagRow = image+channel*imgDIM*2;
    float* curOutputRow = output;
    for(int i=0; i<steadyStateNum; i++)
    {
        steadyOne2Nine(curImagRow, kernel, curOutputRow, imgDIM, kDIM, channel);
        curImagRow+=channel*imgDIM;
        curOutputRow+=outDim;
    }
    // last part...
}
*/

inline float* getKernelPoint(float* kernels, int knlWidth, int knlWInd, int knlHInd, int channel, int numKnls)
{
    return kernels+knlHInd*knlWidth*numKnls*channel+knlWInd*numKnls*channel;
}
inline float* getOutputPoint(float* output, int outputWidth, int heightOffset, int widthOffset, int numKnls)
{
    return output+heightOffset*outputWidth*numKnls+widthOffset*numKnls;
}

float goldChannelReduction(float* imgChannel, float* curKernelEle, int channel)
{
    float final =0.0;
    for(int cInd=0; cInd<channel; cInd++)
    {
        final+= imgChannel[cInd]*curKernelEle[cInd];

    }
    return final;
}

inline float channelReduction(float* imgChannel, float* curKernelEle, int channel )
{
#ifdef USENEON
    float final2 =0.0;
  #ifdef CHANNELUNROLLFACTOR
    int batch = channel/CHANNELUNROLLFACTOR;
    float* curI = imgChannel;
    float* curK = curKernelEle;
    for(int bInd = 0; bInd<batch; bInd++)
    {
    #if CHANNELUNROLLFACTOR == 4
        float32x4_t input1 = vld1q_f32(curI);
        float32x4_t kernel1 = vld1q_f32(curK);
        float32x4_t result1 = vmulq_f32(kernel1,input1);
        float add2Final =vaddvq_f32(result1);
        final2+= add2Final;
        curI+=4;
        curK+=4;
    #elif CHANNELUNROLLFACTOR == 8
        float32x4_t input1 = vld1q_f32(curI);
        float32x4_t kernel1 = vld1q_f32(curK);

        float32x4_t input2 = vld1q_f32(curI+4);
        float32x4_t kernel2 = vld1q_f32(curK+4);

        float32x4_t result1 = vmulq_f32(kernel1,input1);
        float32x4_t result2 = vmlaq_f32(result1,kernel2,input2);

        float add2Final =vaddvq_f32(result2);
        final2+= add2Final;
        curI+=8;
        curK+=8;
    #elif CHANNELUNROLLFACTOR == 16
        float32x4_t input1 = vld1q_f32(curI);
        float32x4_t kernel1 = vld1q_f32(curK);

        float32x4_t input2 = vld1q_f32(curI+4);
        float32x4_t kernel2 = vld1q_f32(curK+4);

        float32x4_t input3 = vld1q_f32(curI+8);
        float32x4_t kernel3 = vld1q_f32(curK+8);


        float32x4_t input4 = vld1q_f32(curI+12);
        float32x4_t kernel4 = vld1q_f32(curK+12);

        float32x4_t result1 = vmulq_f32(kernel1,input1);
        float32x4_t result2 = vmlaq_f32(result1,kernel2,input2);
        float32x4_t result3 = vmlaq_f32(result2,kernel3,input3);
        float32x4_t result4 = vmlaq_f32(result3,kernel4,input4);

        float add2Final =vaddvq_f32(result4);
        final2+=add2Final;
        curI+=16;
        curK+=16;
    #endif
    }
  #endif
    // last part after the batch
  #if CHANNELUNROLLFACTOR !=4 && CHANNELUNROLLFACTOR!=8 && CHANNELUNROLLFACTOR != 16
    final2 = goldChannelReduction(imgChannel, curKernelEle, channel);
  #else
    int doneBatchOffset = CHANNELUNROLLFACTOR*batch;
    int remainingChannel = channel%CHANNELUNROLLFACTOR;
    final2+=goldChannelReduction(imgChannel+doneBatchOffset, curKernelEle+doneBatchOffset, remainingChannel);
  #endif

#else
    float final2 = goldChannelReduction(imgChannel, curKernelEle, channel);
#endif

#ifdef DEBUGNEON
    float final1 = goldChannelReduction(imgChannel, curKernelEle, channel);
    if( (-0.1>final1-final2) || (final1-final2>0.1))
    {
        if(made<2) {
            for (int cInd = 0; cInd < channel; cInd++) {
                rtStringStream << imgChannel[cInd] << "|";
            }
            rtStringStream << "\n";
            for (int cInd = 0; cInd < channel; cInd++) {
                rtStringStream << curKernelEle[cInd] << "|";
            }
            rtStringStream << "\n";
            rtStringStream << final1 << " vs " << final2 << "\n";
            made++;
        }
    }
#endif

    return final2;
}

// this is the dot product without using neon intrinsics
// generated multiple output, each for one output channel
void numericcal_MultiFilterDotProduct(float* kernelPoint, float* inputPoint, float* outputPoint, int channel, int numKnls)
{
    float* curOutputPoint = outputPoint;
    float* curKernelPoint = kernelPoint;
    for(int kInd = 0; kInd < numKnls; kInd++)
    {
        //float* curOutputPoint = outputPoint+kInd;
        //float* curKernelPoint = kernelPoint+kInd*channel;
        float result=channelReduction(inputPoint, curKernelPoint, channel );
        *curOutputPoint+=result;
        curOutputPoint+=1;
        curKernelPoint+=channel;
    }
}

/*

// the bottom most row has beginHeight 2 (knlHeight-1), endHeight 3
// the second last row has beginHeight 1 (knlHeight-2), endHeight 3
void numericcal_ChannelMajorBottomRow(float* curImageRow, int imgWidth, int channel,
                        float* kernels, int numKnls, int knlHeight, int knlWidth,
                        float* curOutputRow, int outputWidth,
                        int beginHeight, int endHeight)
{
    // the left knlWidth-1 columns are special
    for(int leftSColInd = 0; leftSColInd < knlWidth-1; leftSColInd++)
    {
        float* curInPoint = curImageRow + leftSColInd*channel;
        // the first in pixel would affect 'knlHeight' output pixels
        // -- by as much as *kernels[0][0] --- *kernels[knlHeight-1][0] respectively
        // the second one would affect 2*knlHeight
        for(int curFanOutWInd = 0; curFanOutWInd < leftSColInd+1; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                // the output to goto will be
                float* curOuputPoint = getOutputPoint(curOutputRow,outputWidth,knlHeight-1-curFanOutHInd, leftSColInd-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }

    }
    int steadyStateWidth = imgWidth-2*(knlWidth-1);
    float* curSteadyImageRowStart = curImageRow+(knlWidth-1)*channel;
    for(int steadyWInd = 0; steadyWInd<steadyStateWidth; steadyWInd++)
    {
        // for the ones in the middle
        float* curInPoint = curSteadyImageRowStart + steadyWInd*channel;
        float* curOutputBase = curOutputRow+steadyWInd*numKnls;

        for(int curFanOutWInd = 0; curFanOutWInd < knlWInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = 0; curFanOutHInd < knlHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                // the output to goto will be
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, knlWInd-1-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }
    }

    // the right knlWidth-1 columns are also special
    float* rightSColImageStart = curImageRow+(imgWidth-(knlWidth-1))*channel;
    for(int rightSColInd = 0; rightSColInd < knlWidth-1; rightSColInd++)
    {
        float* curInPoint = rightSColImageStart + rightSColInd*channel;
        float* curOutputBase = curOutputRow+steadyStateWidth*numKnls;
        for(int curFanOutWInd = 0; curFanOutWInd < knlWidth-1-rightSColInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = 0; curFanOutHInd < knlHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, knlWidth-1-curFanOutWInd, curFanOutHInd, channel, numKnls);
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, rightSColInd+curFanOutWInd ,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);

            }
        }
    }

}

void numericcal_ChannelMajorSteadyRow(float* curImageRow, int imgWidth, int channel,
                        float* kernels, int numKnls, int knlHeight, int knlWidth,
                        float* curOutputRow, int outputWidth)
{
    // the left knlWidth-1 columns are special
    for(int leftSColInd = 0; leftSColInd < knlWidth-1; leftSColInd++)
    {
        float* curInPoint = curImageRow + leftSColInd*channel;
        // the first in pixel would affect 'knlHeight' output pixels
        // -- by as much as *kernels[0][0] --- *kernels[knlHeight-1][0] respectively
        // the second one would affect 2*knlHeight
        for(int curFanOutWInd = 0; curFanOutWInd < leftSColInd+1; curFanOutWInd++)
        {
            for(int curFanOutHInd = 0; curFanOutHInd < knlHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                // the output to goto will be
                float* curOuputPoint = getOutputPoint(curOutputRow,outputWidth,knlHeight-1-curFanOutHInd, leftSColInd-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }

    }
    int steadyStateWidth = imgWidth-2*(knlWidth-1);
    float* curSteadyImageRowStart = curImageRow+(knlWidth-1)*channel;
    for(int steadyWInd = 0; steadyWInd<steadyStateWidth; steadyWInd++)
    {
        // for the ones in the middle
        float* curInPoint = curSteadyImageRowStart + steadyWInd*channel;
        float* curOutputBase = curOutputRow+steadyWInd*numKnls;

        for(int curFanOutWInd = 0; curFanOutWInd < knlWInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = 0; curFanOutHInd < knlHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                // the output to goto will be
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, knlWInd-1-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }
    }

    // the right knlWidth-1 columns are also special
    float* rightSColImageStart = curImageRow+(imgWidth-(knlWidth-1))*channel;
    for(int rightSColInd = 0; rightSColInd < knlWidth-1; rightSColInd++)
    {
        float* curInPoint = rightSColImageStart + rightSColInd*channel;
        float* curOutputBase = curOutputRow+steadyStateWidth*numKnls;
        for(int curFanOutWInd = 0; curFanOutWInd < knlWidth-1-rightSColInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = 0; curFanOutHInd < knlHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, knlWidth-1-curFanOutWInd, curFanOutHInd, channel, numKnls);
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, rightSColInd+curFanOutWInd ,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);

            }
        }
    }

}
// for row one, begin height is 0, end height is 1, outputHOffset is -2 (knlHeight-1)
// for row two, begin height is 0, end height is 2, outputHOffset is -1 (knlHeight-2)
void numericcal_ChannelMajorTopRows(float* curImageRow, int imgWidth, int channel,
                        float* kernels, int numKnls, int knlHeight, int knlWidth,
                        float* curOutput, int outputWidth,
                        int beginHeight, int endHeight, int outputHOffset)
{
    // curOutput is the first row....but it should actually be -1st or -2nd row
    float* curOutputRow = getOutputPoint(curOutput,outputWidth,outputHOffset,0,numKnls);
    // the left knlWidth-1 columns are special
    for(int leftSColInd = 0; leftSColInd < knlWidth-1; leftSColInd++)
    {
        float* curInPoint = curImageRow + leftSColInd*channel;
        // the first in pixel would affect 'knlHeight' output pixels
        // -- by as much as *kernels[0][0] --- *kernels[knlHeight-1][0] respectively
        // the second one would affect 2*knlHeight
        for(int curFanOutWInd = 0; curFanOutWInd < leftSColInd+1; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                // the output to goto will be
                float* curOuputPoint = getOutputPoint(curOutputRow,outputWidth,knlHeight-1-curFanOutHInd, leftSColInd-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }

    }
    int steadyStateWidth = imgWidth-2*(knlWidth-1);
    float* curSteadyImageRowStart = curImageRow+(knlWidth-1)*channel;
    for(int steadyWInd = 0; steadyWInd<steadyStateWidth; steadyWInd++)
    {
        // for the ones in the middle
        float* curInPoint = curSteadyImageRowStart + steadyWInd*channel;
        float* curOutputBase = curOutputRow+steadyWInd*numKnls;

        for(int curFanOutWInd = 0; curFanOutWInd < knlWInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                // the output to goto will be
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, knlWInd-1-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }
    }

    // the right knlWidth-1 columns are also special
    float* rightSColImageStart = curImageRow+(imgWidth-(knlWidth-1))*channel;
    for(int rightSColInd = 0; rightSColInd < knlWidth-1; rightSColInd++)
    {
        float* curInPoint = rightSColImageStart + rightSColInd*channel;
        float* curOutputBase = curOutputRow+steadyStateWidth*numKnls;
        for(int curFanOutWInd = 0; curFanOutWInd < knlWidth-1-rightSColInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, knlWidth-1-curFanOutWInd, curFanOutHInd, channel, numKnls);
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, rightSColInd+curFanOutWInd ,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);

            }
        }
    }

}
*/

// oneRow in the core
void numericcal_ChannelMajorCore(float* curImageRow, int imgWidth, int channel,
                                 float* kernels, int numKnls, int knlHeight, int knlWidth,
                                 float* curOutput, int outputWidth,
                                 int beginHeight, int endHeight, int outputHOffset)
{
    float* curOutputRow = getOutputPoint(curOutput,outputWidth,outputHOffset,0,numKnls);
    for(int leftSColInd = 0; leftSColInd < knlWidth-1; leftSColInd++)
    {
        float* curInPoint = curImageRow + leftSColInd*channel;
        for(int curFanOutWInd = 0; curFanOutWInd < leftSColInd+1; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                float* curOuputPoint = getOutputPoint(curOutputRow,outputWidth,knlHeight-1-curFanOutHInd, leftSColInd-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }

    }
    int steadyStateWidth = imgWidth-2*(knlWidth-1);
    float* curSteadyImageRowStart = curImageRow+(knlWidth-1)*channel;
    for(int steadyWInd = 0; steadyWInd<steadyStateWidth; steadyWInd++)
    {
        float* curInPoint = curSteadyImageRowStart + steadyWInd*channel;
        float* curOutputBase = curOutputRow+steadyWInd*numKnls;

        for(int curFanOutWInd = 0; curFanOutWInd < knlWidth; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, curFanOutWInd, curFanOutHInd, channel, numKnls);
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, knlWidth-1-curFanOutWInd,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);
            }
        }
    }
    // the right knlWidth-1 columns are also special
    float* rightSColImageStart = curImageRow+(imgWidth-(knlWidth-1))*channel;
    for(int rightSColInd = 0; rightSColInd < knlWidth-1; rightSColInd++)
    {
        float* curInPoint = rightSColImageStart + rightSColInd*channel;
        float* curOutputBase = curOutputRow+steadyStateWidth*numKnls;
        for(int curFanOutWInd = 0; curFanOutWInd < knlWidth-1-rightSColInd; curFanOutWInd++)
        {
            for(int curFanOutHInd = beginHeight; curFanOutHInd < endHeight; curFanOutHInd++)
            {
                float* curKernelPoint = getKernelPoint(kernels, knlWidth, knlWidth-1-curFanOutWInd, curFanOutHInd, channel, numKnls);
                float* curOuputPoint = getOutputPoint(curOutputBase,outputWidth,knlHeight-1-curFanOutHInd, rightSColInd+curFanOutWInd ,numKnls);
                numericcal_MultiFilterDotProduct(curKernelPoint,curInPoint,curOuputPoint,channel, numKnls);

            }
        }
    }

}


void numericcal_ChannelMajor(float* inputImage, int imgWidth, int imgHeight, int channel,
                             float* kernels, int numKnls, int knlWidth, int knlHeight,
                             float* output, int outputWidth, int outputHeight)
{
    assert(outputWidth == imgWidth-knlWidth+1);
    assert(outputHeight == imgHeight-knlHeight+1);
    int sizePerImgRow = imgWidth*channel;
    // special treatment for the boundary rows
    for(int topRowInd = 0; topRowInd<knlHeight-1; topRowInd++)
    {
        int outputHOffset = topRowInd-(knlHeight-1);
        numericcal_ChannelMajorCore(inputImage+sizePerImgRow*topRowInd, imgWidth, channel,
                                    kernels, numKnls, knlHeight, knlWidth, output, outputWidth,
                                    0, topRowInd+1, outputHOffset);
    }
    // the core part, the top knlHeight-1 row and the bottom knlHeight-1 row
    // are dealt with separately
    int steadyStateHeight = imgHeight-2*(knlHeight-1);
    float* curImageRow = inputImage+sizePerImgRow*(knlHeight-1);
    float* curOutputRow = output;
    for(int steadyHInd = 0; steadyHInd < steadyStateHeight; steadyHInd++)
    {
        numericcal_ChannelMajorCore(curImageRow, imgWidth, channel,
                                    kernels, numKnls, knlHeight, knlWidth, curOutputRow, outputWidth,
                                    0, knlHeight, 0);
        curImageRow += sizePerImgRow;
        curOutputRow += outputWidth*numKnls;
    }
    // the bottom part
    for(int bottomRowInd = 0; bottomRowInd<knlHeight-1; bottomRowInd++)
    {
        numericcal_ChannelMajorCore(curImageRow, imgWidth, channel,
                                    kernels, numKnls, knlHeight, knlWidth, curOutputRow, outputWidth,
                                    bottomRowInd+1, knlHeight, 0);
        curImageRow += sizePerImgRow;
        curOutputRow += outputWidth*numKnls;
    }


}
/* gold model for row major */
/* dimension of image:
 *  height, channel, width
 *  /



/* gold model for channel major */
/* dimension of image:
 *   height, width, channel
 * dimension of kernels:
 *   height, width, numKernels, channel
 * dimension of output:
 *   height, width, channel
 * */
float goldDotProduct(float* inputPixel, float* kernelPixel, int numPoint)
{
    float result = 0.0;
    for(int k=0; k<numPoint; k++)
    {
        result += inputPixel[k]*kernelPixel[k];
    }
    return result;
}

void goldSinglePointChannelMajor(float* inputOrigin, int imgWidth, int imgHeight,
                                 float* curOutputLocation, float* kernels, int knlWidth, int knlHeight, int numKnls,
                                 int channel)
{

    for(int kHInd = 0; kHInd<knlHeight; kHInd++)
    {
        for(int kWInd=0; kWInd<knlWidth; kWInd++)
        {
            for(int kInd=0; kInd<numKnls; kInd++)
            {
                float* curKernelPoint = kernels+kHInd*knlWidth*numKnls*channel+kWInd*numKnls*channel+kInd*channel;
                float* curInputPoint = inputOrigin+kHInd*imgWidth*channel+kWInd*channel;
                *(curOutputLocation+kInd) += goldDotProduct(curInputPoint, curKernelPoint, channel);
            }
        }
    }
}

void goldChannelMajor(float* inputImage, int imgWidth, int imgHeight, int channel,
                      float* kernels, int numKnls, int knlWidth, int knlHeight,
                      float* output, int outputWidth, int outputHeight)
{
    assert(outputWidth == imgWidth-knlWidth+1);
    assert(outputHeight == imgHeight-knlHeight+1);
    for (int outHInd = 0; outHInd < outputHeight; outHInd++)
        for (int outWInd = 0; outWInd < outputWidth; outWInd++){
            float *curOutputLocation =
                    output + outHInd * outputWidth * numKnls + outWInd * numKnls;
            float *inputOrigin = inputImage + outHInd * imgWidth * channel + outWInd * channel;
            goldSinglePointChannelMajor(inputOrigin,imgWidth,imgHeight,curOutputLocation,kernels,knlWidth, knlHeight, numKnls, channel);
        }
}
/* end of golden model for channel major */
/* compare two array */
float l1DistanceOutput(float* arr1, float* arr2, int numElement)
{
    float l1D = 0.0;
    for(int eleInd = 0; eleInd < numElement; eleInd++)
    {
        l1D += (arr1[eleInd]-arr2[eleInd] >=0.0 ? (arr1[eleInd]-arr2[eleInd]):(arr2[eleInd]-arr1[eleInd]));
    }
    return l1D;
}

float norml1DistanceOutput(float* arr1, float* arr2, int numElement)
{
    float l1D = 0.0;
    for(int eleInd = 0; eleInd < numElement; eleInd++)
    {
        float curEleL1D = (arr1[eleInd]-arr2[eleInd] >=0.0 ? (arr1[eleInd]-arr2[eleInd]):(arr2[eleInd]-arr1[eleInd]));
        if(arr2[eleInd]!=0.0)
            l1D += curEleL1D/arr2[eleInd];
        else
            l1D += curEleL1D;
    }
    return l1D;
}

#ifndef RUNDEVICE
/* tmp main to show the result of convolution */
#define KW 3
#define KH 3
#define IMGW 4
#define IMGH 4
#define CHAN 2
#define NK 3

void dumpArray(float* arr, int height, int width, int num, int channel)
{
    for(int hInd = 0; hInd < height; hInd++)
    {
        std::cout<<"[\t";
        for(int wInd = 0; wInd < width; wInd++)
        {
            for(int kInd = 0; kInd < num; kInd++)
            {
                if(kInd!=0)
                    std::cout<<" |";
                for(int cInd = 0; cInd < channel; cInd++)
                {
                    std::cout<<arr[hInd*width*num*channel+wInd*num*channel+kInd*channel+cInd]<<" ";
                }
            }
            if(wInd!=width-1)
                std::cout<<",";
        }
        std::cout <<"\t]\n";
    }
}

int main()
{
    int imgWidth = IMGW;
    int imgHeight = IMGH;
    int channel = CHAN;
    int numKnls = NK;
    int knlWidth = KW;
    int knlHeight = KH;
    float* inputImage = allocateAndPopulate( 1, imgWidth, imgHeight,channel,FILLPOSITION);
    float* kernels = allocateAndPopulate(numKnls, knlWidth,knlHeight,channel,FILLPOSITION);
    int outputWidth = imgWidth-knlWidth+1;
    int outputHeight = imgHeight-knlHeight+1;
    float* goldOutput = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);
    goldChannelMajor(inputImage,imgWidth,imgHeight,channel,kernels,numKnls, knlWidth,knlHeight,goldOutput,outputWidth,outputHeight);
    float* numericcalOutput = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);
    numericcal_ChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls, knlWidth, knlHeight, numericcalOutput, outputWidth,
                            outputHeight);

#ifdef ELABORATECOMP
    std::cout << "image input:\n";
    dumpArray(inputImage,imgHeight,imgWidth,1,channel);
    std::cout << "\nkernel input:\n";
    dumpArray(kernels,knlHeight,knlWidth,numKnls,channel);
    // there is just one output
    std::cout<<"\noutput:\n";
    dumpArray(goldOutput, outputHeight, outputWidth, 1, numKnls);
    std::cout<<"\noutput2:\n";
    dumpArray(numericcalOutput, outputHeight, outputWidth, 1, numKnls);
#endif
    std::cout<<"L1 distance between numericcal and gold output:"<<l1DistanceOutput(goldOutput,numericcalOutput,outputHeight*outputWidth*numKnls);


    deallocate(inputImage);
    deallocate(kernels);
    deallocate(goldOutput);
    deallocate(numericcalOutput);

}
#endif



#ifdef RUNDEVICE
extern "C"
{
/*
    jstring
    Java_com_numericcal_convolutionbenchmark_LaunchBenchmark_launchConv(
            JNIEnv *env,
            jobject,
            int repeat, int imgWidth, int imgHeight,
            int knlWidth, int knlHeight, int channel,
            int numKnls)
    {
        std::stringstream rtStringStream;
        float* inputImage = allocateAndPopulate( 1, imgWidth, imgHeight,channel,FILLRAND);
        float* kernels = allocateAndPopulate(numKnls, knlWidth,knlHeight,channel,FILLRAND);
        int outputWidth = imgWidth-knlWidth+1;
        int outputHeight = imgHeight-knlHeight+1;
        float* output = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);
        struct timespec timecount;
        timecount = timer_start();
        for(int repInd = 0; repInd < repeat; repInd++)
        {
            sconv_mm(false, inputImage, imgWidth, imgHeight, channel,
                     kernels, numKnls, knlWidth, knlHeight, 0, 0,
                     1, 1, output, outputWidth, outputHeight);
        }
        long long timeSpent = timer_end(timecount);
        deallocate(inputImage);
        deallocate(kernels);
        deallocate(output);
        rtStringStream<<timeSpent/1000000<< "ms";
        return env->NewStringUTF(rtStringStream.str().c_str());
    }*/

    jstring
    Java_com_numericcal_convolutionbenchmark_LaunchBenchmark_launchConvNumericcal(
            JNIEnv *env,
            jobject,
            int repeat, int imgWidth, int imgHeight,
            int knlWidth, int knlHeight, int channel,
            int numKnls)
    {
        rtStringStream.str("");
        made=false;
        //std::stringstream rtStringStream;
        int outputWidth = imgWidth-knlWidth+1;
        int outputHeight = imgHeight-knlHeight+1;

        float* inputImage = allocateAndPopulate( 1, imgWidth, imgHeight,channel,FILLPOSITION);
        float* kernels = allocateAndPopulate(numKnls, knlWidth,knlHeight,channel, FILLPOSITION);
        float* numericcalOutput = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);

        struct timespec timecount;
        timecount = timer_start();
        for(int repInd = 0; repInd < repeat; repInd++)
        {
            numericcal_ChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls, knlWidth,
                                    knlHeight, numericcalOutput, outputWidth,
                                    outputHeight);
        }
        long long timeSpent = timer_end(timecount);
        // now perform the comparison with the golden model
        float* goldenOutput = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);
        repopulate(numericcalOutput,numKnls,outputWidth,outputHeight,1,FILLZERO);

        numericcal_ChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls, knlWidth,
                                    knlHeight, numericcalOutput, outputWidth,
                                    outputHeight);
        goldChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls, knlWidth, knlHeight,
                goldenOutput, outputWidth, outputHeight);
        float norml1dis = norml1DistanceOutput(numericcalOutput,goldenOutput,outputWidth*outputHeight*numKnls);
        float l1dis = l1DistanceOutput(numericcalOutput,goldenOutput,outputWidth*outputHeight*numKnls);
        timecount = timer_start();
        for(int repInd = 0; repInd < repeat; repInd++)
        {
            sconv_mm(false, inputImage, imgWidth, imgHeight, channel,
                     kernels, numKnls, knlWidth, knlHeight, 0, 0,
                     1, 1, numericcalOutput, outputWidth, outputHeight);
        }
        long long timeSpent2 = timer_end(timecount);
        deallocate(inputImage);
        deallocate(kernels);
        deallocate(numericcalOutput);
        deallocate(goldenOutput);
        rtStringStream<<timeSpent/1000000<< "ms v.s. QSML"<<timeSpent2/1000000<<", L1 dist to Golden "<<norml1dis<<"(normalized),"<<l1dis<<"(absolute)\n";
        return env->NewStringUTF(rtStringStream.str().c_str());
    }
    void
    Java_com_numericcal_convolutionbenchmark_MainActivity_launchConvNumericcalDummy(
            JNIEnv *env,
            jobject)
    {
        int repeat=10;
        int imgWidth=512;
        int imgHeight=512;
        int knlWidth=3;
        int knlHeight=3;
        int channel_upper=64;
        int channel_lower=1;
        int numKnls=1;

        //LOGD("Hello world");
        int outputWidth = imgWidth-knlWidth+1;
        int outputHeight = imgHeight-knlHeight+1;

        float* inputImage = allocateAndPopulate( 1, imgWidth, imgHeight,channel_upper,FILLPOSITION);
        float* kernels = allocateAndPopulate(numKnls, knlWidth,knlHeight,channel_upper, FILLPOSITION);
        float* numericcalOutput = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);
        float *goldenOutput = allocateAndPopulate(numKnls, outputWidth, outputHeight, 1, FILLZERO);
    for(int channel=channel_upper; channel>=channel_lower; channel--) {
        struct timespec timecount;
        timecount = timer_start();
        for (int repInd = 0; repInd < repeat; repInd++) {
            numericcal_ChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls,
                                    knlWidth,
                                    knlHeight, numericcalOutput, outputWidth,
                                    outputHeight);
        }
        long long timeSpent = timer_end(timecount);
        // now perform the comparison with the golden model

        repopulate(numericcalOutput, numKnls, outputWidth, outputHeight, 1, FILLZERO);

        numericcal_ChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls,
                                knlWidth,
                                knlHeight, numericcalOutput, outputWidth,
                                outputHeight);
        goldChannelMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls, knlWidth,
                         knlHeight,
                         goldenOutput, outputWidth, outputHeight);
        float norml1dis = norml1DistanceOutput(numericcalOutput, goldenOutput,
                                               outputWidth * outputHeight * numKnls);
        float l1dis = l1DistanceOutput(numericcalOutput, goldenOutput,
                                       outputWidth * outputHeight * numKnls);
        repopulate(numericcalOutput, numKnls, outputWidth, outputHeight, 1, FILLZERO);
        repopulate(goldenOutput, numKnls, outputWidth, outputHeight, 1, FILLZERO);
        timecount = timer_start();
        for (int repInd = 0; repInd < repeat; repInd++) {
            sconv_mm(false, inputImage, imgWidth, imgHeight, channel,
                     kernels, numKnls, knlWidth, knlHeight, 0, 0,
                     1, 1, numericcalOutput, outputWidth, outputHeight);
        }
        long long timeSpent2 = timer_end(timecount);
        LOGD("%d channel: %lld ms v.s. QSML %lld, L1 dist to Golden %f (normalized), %f (absolute)\n", channel, timeSpent/1000000, timeSpent2/1000000,norml1dis,l1dis);
    }
        deallocate(inputImage);
        deallocate(kernels);
        deallocate(numericcalOutput);
        deallocate(goldenOutput);

        //rtStringStream<<timeSpent/1000000<< "ms v.s. QSML"<<timeSpent2/1000000<<", L1 dist to Golden "<<norml1dis<<"(normalized),"<<l1dis<<"(absolute)\n";
    }

}
#endif
