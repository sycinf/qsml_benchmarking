#define RUNDEVICE
#define MAXELE 20
#define MAJOR CHANNEL
//#define DEBUGNEON
#define USENEON
#define CHANNELUNROLLFACTOR 16

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
/* dimension of image:
 *  height, channel, width
 * dimension of filter
 *  height,channel,numKernels,width
 * dimension of output
 *  height, channel, width
 */
void oneEleOneRow1(float* inputRow, int numMul, float* outputRow, float* element)
{
    float v = *element;
    for(int i=0; i<numMul; i++)
    {
        float m = *(inputRow+i);
        *(outputRow+i) += m*v;

    }

}


inline void oneEleOneRow1_8(float* inputRow, int numMul, float* outputRow, float* element)
{
    int num8 = numMul/8;
    float32x4_t eleDup = vld1q_dup_f32(element);
    float* curIn = inputRow;
    float* nextIn = inputRow+8;
    float* curOut = outputRow;
    float* nextOut = outputRow+8;
    for(int i=0; i<num8; i++)
    {
        float* curOut1 = curOut;
        float32x4_t input1 = vld1q_f32(curIn);
        float32x4_t output1 =  vld1q_f32(curOut1);
        float* curOut2 = curOut+4;
        float32x4_t input2 = vld1q_f32(curIn+4);
        float32x4_t output2 =  vld1q_f32(curOut2);


        float32x4_t result1 = vmlaq_f32(output1,eleDup,input1);
        float32x4_t result2 = vmlaq_f32(output2,eleDup,input2);



        vst1q_f32(curOut1,result1);
        vst1q_f32(curOut2,result2);


        curIn = nextIn;
        nextIn += 8;
        curOut = nextOut;
        nextOut += 8;
    }
    oneEleOneRow1(inputRow+num8*8,numMul%8,outputRow+num8*8,element);
    //int  doneI = numQ*28;
    //oneEleOneRow(inputRow+doneI,numMul-doneI,outputRow+doneI,element);
    //for(int i=numQ*28; i<numMul; i++)
    //{
    //    oneEleOneRow(inputRow+i,numMul-i,outputRow+i,element);

    //}

}


inline void oneEleOneRow1_16(float* inputRow, int numMul, float* outputRow, float* element)
{
    int num16 = numMul/16;
    float32x4_t eleDup = vld1q_dup_f32(element);
    float* curIn = inputRow;
    float* nextIn = inputRow+16;
    float* curOut = outputRow;
    float* nextOut = outputRow+16;
    for(int i=0; i<num16; i++)
    {
        float* curOut1 = curOut;
        float32x4_t input1 = vld1q_f32(curIn);
        float32x4_t output1 =  vld1q_f32(curOut1);
        float* curOut2 = curOut+4;
        float32x4_t input2 = vld1q_f32(curIn+4);
        float32x4_t output2 =  vld1q_f32(curOut2);
        float* curOut3 = curOut+8;
        float32x4_t input3 = vld1q_f32(curIn+8);
        float32x4_t output3 =  vld1q_f32(curOut3);
        float* curOut4 = curOut+12;
        float32x4_t input4 = vld1q_f32(curIn+12);
        float32x4_t output4 =  vld1q_f32(curOut4);

        float32x4_t result1 = vmlaq_f32(output1,eleDup,input1);
        float32x4_t result2 = vmlaq_f32(output2,eleDup,input2);
        float32x4_t result3 = vmlaq_f32(output3,eleDup,input3);
        float32x4_t result4 = vmlaq_f32(output4,eleDup,input4);


        vst1q_f32(curOut1,result1);
        vst1q_f32(curOut2,result2);
        vst1q_f32(curOut3,result3);
        vst1q_f32(curOut4,result4);

        curIn = nextIn;
        nextIn += 16;
        curOut = nextOut;
        nextOut += 16;
    }
    oneEleOneRow1_8(inputRow+num16*16,numMul%16,outputRow+num16*16,element);
    //int  doneI = numQ*28;
    //oneEleOneRow(inputRow+doneI,numMul-doneI,outputRow+doneI,element);
    //for(int i=numQ*28; i<numMul; i++)
    //{
    //    oneEleOneRow(inputRow+i,numMul-i,outputRow+i,element);

    //}

}


void oneEleOneRow1_32(float* inputRow, int numMul, float* outputRow, float* element)
{
    int num32 = numMul/32;
    float32x4_t eleDup = vld1q_dup_f32(element);
    float* curIn = inputRow;
    float* nextIn = inputRow+32;
    float* curOut = outputRow;
    float* nextOut = outputRow+32;
    for(int i=0; i<num32; i++)
    {
        /*__asm__ volatile(
        "mov %0, %[nIter]\n\t"
        "PRFM PLDL1KEEP, [%0]"
        :
        : [nIter] "r" (nextIn)
        );
        __asm__ volatile(
        "mov %0, %[nIter]\n\t"
        "PRFM PLDL1STRM, [%0]"
        :
        : [nIter] "r" (nextOut)
        );*/

        float* curOut1 = curOut;
        float32x4_t input1 = vld1q_f32(curIn);
        float32x4_t output1 =  vld1q_f32(curOut1);
        float* curOut2 = curOut+4;
        float32x4_t input2 = vld1q_f32(curIn+4);
        float32x4_t output2 =  vld1q_f32(curOut2);
        float* curOut3 = curOut+8;
        float32x4_t input3 = vld1q_f32(curIn+8);
        float32x4_t output3 =  vld1q_f32(curOut3);
        float* curOut4 = curOut+12;
        float32x4_t input4 = vld1q_f32(curIn+12);
        float32x4_t output4 =  vld1q_f32(curOut4);

        float* curOut5 = curOut+16;
        float32x4_t input5 = vld1q_f32(curIn+16);
        float32x4_t output5 =  vld1q_f32(curOut5);
        float* curOut6 = curOut+20;
        float32x4_t input6 = vld1q_f32(curIn+20);
        float32x4_t output6 =  vld1q_f32(curOut6);
        float* curOut7 = curOut+24;
        float32x4_t input7 = vld1q_f32(curIn+24);
        float32x4_t output7 =  vld1q_f32(curOut7);
        float* curOut8 = curOut+28;
        float32x4_t input8 = vld1q_f32(curIn+28);
        float32x4_t output8 =  vld1q_f32(curOut8);

        float32x4_t result1 = vmlaq_f32(output1,eleDup,input1);
        float32x4_t result2 = vmlaq_f32(output2,eleDup,input2);
        float32x4_t result3 = vmlaq_f32(output3,eleDup,input3);
        float32x4_t result4 = vmlaq_f32(output4,eleDup,input4);
        float32x4_t result5 = vmlaq_f32(output5,eleDup,input5);
        float32x4_t result6 = vmlaq_f32(output6,eleDup,input6);
        float32x4_t result7 = vmlaq_f32(output7,eleDup,input7);
        float32x4_t result8 = vmlaq_f32(output8,eleDup,input8);

        vst1q_f32(curOut1,result1);
        vst1q_f32(curOut2,result2);
        vst1q_f32(curOut3,result3);
        vst1q_f32(curOut4,result4);
        vst1q_f32(curOut5,result5);
        vst1q_f32(curOut6,result6);
        vst1q_f32(curOut7,result7);
        vst1q_f32(curOut8,result8);
        curIn = nextIn;
        nextIn += 32;
        curOut = nextOut;
        nextOut += 32;
    }
    oneEleOneRow1_16(inputRow+num32*32,numMul%32,outputRow+num32*32,element);
    //int  doneI = numQ*28;
    //oneEleOneRow(inputRow+doneI,numMul-doneI,outputRow+doneI,element);
    //for(int i=numQ*28; i<numMul; i++)
    //{
    //    oneEleOneRow(inputRow+i,numMul-i,outputRow+i,element);

    //}

}



inline void oneEleOneRow1_28(float* inputRow, int numMul, float* outputRow, float* element)
{
    int num28 = numMul/28;
    float32x4_t eleDup = vld1q_dup_f32(element);
    float* curIn = inputRow;
    float* nextIn = inputRow+28;
    float* curOut = outputRow;
    float* nextOut = outputRow+28;
    for(int i=0; i<num28; i++)
    {
        /*__asm__ volatile(
        "mov %0, %[nIter]\n\t"
        "PRFM PLDL1KEEP, [%0]"
        :
        : [nIter] "r" (nextIn)
        );
        __asm__ volatile(
        "mov %0, %[nIter]\n\t"
        "PRFM PLDL1STRM, [%0]"
        :
        : [nIter] "r" (nextOut)
        );*/

        float* curOut1 = curOut;
        float32x4_t input1 = vld1q_f32(curIn);
        float32x4_t output1 =  vld1q_f32(curOut1);
        float* curOut2 = curOut+4;
        float32x4_t input2 = vld1q_f32(curIn+4);
        float32x4_t output2 =  vld1q_f32(curOut2);
        float* curOut3 = curOut+8;
        float32x4_t input3 = vld1q_f32(curIn+8);
        float32x4_t output3 =  vld1q_f32(curOut3);
        float* curOut4 = curOut+12;
        float32x4_t input4 = vld1q_f32(curIn+12);
        float32x4_t output4 =  vld1q_f32(curOut4);

        float* curOut5 = curOut+16;
        float32x4_t input5 = vld1q_f32(curIn+16);
        float32x4_t output5 =  vld1q_f32(curOut5);
        float* curOut6 = curOut+20;
        float32x4_t input6 = vld1q_f32(curIn+20);
        float32x4_t output6 =  vld1q_f32(curOut6);
        float* curOut7 = curOut+24;
        float32x4_t input7 = vld1q_f32(curIn+24);
        float32x4_t output7 =  vld1q_f32(curOut7);


        float32x4_t result1 = vmlaq_f32(output1,eleDup,input1);
        float32x4_t result2 = vmlaq_f32(output2,eleDup,input2);
        float32x4_t result3 = vmlaq_f32(output3,eleDup,input3);
        float32x4_t result4 = vmlaq_f32(output4,eleDup,input4);
        float32x4_t result5 = vmlaq_f32(output5,eleDup,input5);
        float32x4_t result6 = vmlaq_f32(output6,eleDup,input6);
        float32x4_t result7 = vmlaq_f32(output7,eleDup,input7);


        vst1q_f32(curOut1,result1);
        vst1q_f32(curOut2,result2);
        vst1q_f32(curOut3,result3);
        vst1q_f32(curOut4,result4);
        vst1q_f32(curOut5,result5);
        vst1q_f32(curOut6,result6);
        vst1q_f32(curOut7,result7);

        curIn = nextIn;
        nextIn += 28;
        curOut = nextOut;
        nextOut += 28;
    }
    oneEleOneRow1_16(inputRow+num28*28,numMul%28,outputRow+num28*28,element);
    //int  doneI = numQ*28;
    //oneEleOneRow(inputRow+doneI,numMul-doneI,outputRow+doneI,element);
    //for(int i=numQ*28; i<numMul; i++)
    //{
    //    oneEleOneRow(inputRow+i,numMul-i,outputRow+i,element);

    //}

}




/* dimension of image:
 *  height, channel, width
 * dimension of filter
 *  height,channel,numKernels,width
 * dimension of output
 *  height, channel, width
 */
void numericcal_RowMajor(float* inputImage, int imgWidth, int imgHeight, int channel,
                 float* kernels, int numKnls, int knlWidth, int knlHeight,
                 float* output, int outputWidth, int outputHeight)
{
    // only support this configuration for now
    assert(knlWidth == 3 && knlHeight ==3 && numKnls ==1);


    float* output1Row = output;
    float* input1Row = inputImage;
    //int rowMul = outputWidth;
    int offsetIn = 0;
    int offsetOut = 0;

    int knlPerChannelSize = knlWidth*numKnls;
    int imgRowSize = channel*imgWidth;
    float* curInputRow = input1Row;
    float* curKnlChannel = kernels;
    // first row
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        oneEleOneRow1_28(curInputRow, outputWidth, output1Row, curKnlChannel+offsetChannel);
        oneEleOneRow1_28(curInputRow+1, outputWidth, output1Row, curKnlChannel+offsetChannel+1);
        oneEleOneRow1_28(curInputRow+2, outputWidth, output1Row, curKnlChannel+offsetChannel+2);
        curInputRow+=imgWidth;
    }
    int knlPerRowSize = knlPerChannelSize*channel;
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        oneEleOneRow1_28(curInputRow, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ offsetChannel/*kernel[3]*/);
        oneEleOneRow1_28(curInputRow + 1, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ offsetChannel+1/*kernel[4]*/);
        oneEleOneRow1_28(curInputRow + 2, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ offsetChannel+2/*kernel[5]*/);

        oneEleOneRow1_28(curInputRow, outputWidth, output1Row + outputWidth, curKnlChannel+offsetChannel);
        oneEleOneRow1_28(curInputRow + 1, outputWidth, output1Row + outputWidth, curKnlChannel+offsetChannel+1);
        oneEleOneRow1_28(curInputRow + 2, outputWidth, output1Row + outputWidth, curKnlChannel+offsetChannel+2);
        curInputRow+=imgWidth;
    }

    int steadyStateRowNum = imgHeight-2*(knlHeight-1);
    //steady state
    for(int k=0; k< steadyStateRowNum ; k++) {

        for(int cInd = 0; cInd < channel; cInd++) {
            int offsetChannel = cInd*knlPerChannelSize;
            oneEleOneRow1_28(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);
            oneEleOneRow1_28(curInputRow + 1, outputWidth, output1Row, curKnlChannel + 2*knlPerRowSize+ offsetChannel+ 1);
            oneEleOneRow1_28(curInputRow + 2, outputWidth, output1Row, curKnlChannel + 2*knlPerRowSize+ offsetChannel+ 2);

            oneEleOneRow1_28(curInputRow, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ offsetChannel );
            oneEleOneRow1_28(curInputRow + 1, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ offsetChannel+1);
            oneEleOneRow1_28(curInputRow + 2, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ offsetChannel+2 );

            oneEleOneRow1_28(curInputRow, outputWidth, output1Row + outputWidth*2, curKnlChannel+ offsetChannel );
            oneEleOneRow1_28(curInputRow + 1, outputWidth, output1Row + outputWidth*2,curKnlChannel+ offsetChannel+1 );
            oneEleOneRow1_28(curInputRow + 2, outputWidth, output1Row + outputWidth*2,curKnlChannel+ offsetChannel+2 );
            curInputRow+=imgWidth;
        }
        output1Row+=(outputWidth*numKnls);
    }
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        oneEleOneRow1_28(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);
        oneEleOneRow1_28(curInputRow + 1, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel+1);
        oneEleOneRow1_28(curInputRow + 2, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel+2);

        oneEleOneRow1_28(curInputRow , outputWidth, output1Row+outputWidth, curKnlChannel+ knlPerRowSize+ offsetChannel);
        oneEleOneRow1_28(curInputRow +1, outputWidth, output1Row+outputWidth, curKnlChannel+ knlPerRowSize+ offsetChannel+1);
        oneEleOneRow1_28(curInputRow +2, outputWidth, output1Row+outputWidth, curKnlChannel+ knlPerRowSize+ offsetChannel+2);
        curInputRow+=imgWidth;
    }
    output1Row+=(outputWidth*numKnls);

    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd * knlPerChannelSize;
        oneEleOneRow1_28(curInputRow , outputWidth, output1Row , curKnlChannel+ 2*knlPerRowSize+ offsetChannel);
        oneEleOneRow1_28(curInputRow  + 1, outputWidth, output1Row ,curKnlChannel+ 2*knlPerRowSize+ offsetChannel+1);
        oneEleOneRow1_28(curInputRow  + 2, outputWidth, output1Row , curKnlChannel+ 2*knlPerRowSize+ offsetChannel+2);
        curInputRow+=imgWidth;
    }
}

void threeEleThreeRow(float* inputRow, int numMul, float* outputRow, float* element)
{
    //this is sliding window
    // 1. we load whole bunch
    float32x4_t ele0Dup = vld1q_dup_f32(element);
    float32x4_t ele1Dup = vld1q_dup_f32(element+1);
    float32x4_t ele2Dup = vld1q_dup_f32(element+2);
    float* curBatch = inputRow;
    float* curOutBatch = outputRow;
    int num4 = numMul/4;
    for(int i = 0; i < num4; i++ ) {

        float32x4_t input0 = vld1q_f32(curBatch);
        float32x4_t input1 = vld1q_f32(curBatch + 1);
        float32x4_t input2 = vld1q_f32(curBatch + 2);

        float32x4_t output0 = vld1q_f32(curOutBatch);

        float32x4_t result1;
        result1 = vmlaq_f32(output0, ele0Dup, input0);
        result1 = vmlaq_f32(result1, ele1Dup, input1);
        result1 = vmlaq_f32(result1, ele2Dup, input2);
        vst1q_f32(curOutBatch, result1);
        curBatch+=4;
        curOutBatch+=4;
    }

    int done = num4*4;
    curBatch = inputRow+done;
    curOutBatch = outputRow+done;


    float ele0 = *element;
    float ele1 = *(element+1);
    float ele2 = *(element+2);
    for(int i = 0; i < numMul%4; i++)
    {
        float m0 = *(curBatch+i);
        float m1 = *(curBatch+i+1);
        float m2 = *(curBatch+i+2);
        *(curOutBatch+i) += m0*ele0+m1*ele1+m2*ele2;
    }




    /*oneEleOneRow1_28(inputRow, numMul, outputRow, element);
    oneEleOneRow1_28(inputRow+1, numMul, outputRow, element+1);
    oneEleOneRow1_28(inputRow+2, numMul, outputRow, element+2);*/
}


void threeEleThreeRow_2c(float* inputRow, int numMul, float* outputRow, float* element,
                               int inputChannelOffset, int kernelChannelOffset)
{
    //this is sliding window
    // 1. we load whole bunch
    float32x4_t ele0Dup = vld1q_dup_f32(element);
    float32x4_t ele1Dup = vld1q_dup_f32(element+1);
    float32x4_t ele2Dup = vld1q_dup_f32(element+2);

    float32x4_t ele0Dup_c = vld1q_dup_f32(element+kernelChannelOffset);
    float32x4_t ele1Dup_c = vld1q_dup_f32(element+kernelChannelOffset+1);
    float32x4_t ele2Dup_c = vld1q_dup_f32(element+kernelChannelOffset+2);


    float* curBatch = inputRow;
    float* curOutBatch = outputRow;

    float* curBatch_c = inputRow+inputChannelOffset;


    int num4 = numMul/4;
    for(int i = 0; i < num4; i++ ) {
        float32x4_t result1;
        float32x4_t output0 = vld1q_f32(curOutBatch);

        float32x4_t input0 = vld1q_f32(curBatch);
        float32x4_t input1 = vld1q_f32(curBatch + 1);
        float32x4_t input2 = vld1q_f32(curBatch + 2);

        float32x4_t input0_c = vld1q_f32(curBatch_c);
        float32x4_t input1_c = vld1q_f32(curBatch_c + 1);
        float32x4_t input2_c = vld1q_f32(curBatch_c + 2);






        result1 = vmlaq_f32(output0, ele0Dup, input0);
        result1 = vmlaq_f32(result1, ele1Dup, input1);
        result1 = vmlaq_f32(result1, ele2Dup, input2);

        result1 = vmlaq_f32(result1, ele0Dup_c, input0_c);
        result1 = vmlaq_f32(result1, ele1Dup_c, input1_c);
        result1 = vmlaq_f32(result1, ele2Dup_c, input2_c);




        vst1q_f32(curOutBatch, result1);
        curBatch+=4;
        curBatch_c+=4;
        curOutBatch+=4;
    }

    int done = num4*4;
    curBatch = inputRow+done;
    curOutBatch = outputRow+done;


    float ele0 = *element;
    float ele1 = *(element+1);
    float ele2 = *(element+2);
    float ele0_c = *(element+kernelChannelOffset);
    float ele1_c = *(element+kernelChannelOffset+1);
    float ele2_c = *(element+kernelChannelOffset+2);
    for(int i = 0; i < numMul%4; i++)
    {
        float m0 = *(curBatch+i);
        float m1 = *(curBatch+i+1);
        float m2 = *(curBatch+i+2);

        float m0_c = *(curBatch+inputChannelOffset+i);
        float m1_c = *(curBatch+inputChannelOffset+i+1);
        float m2_c = *(curBatch+inputChannelOffset+i+2);
        *(curOutBatch+i) += m0*ele0+m1*ele1+m2*ele2+m0_c*ele0_c+m1_c*ele1_c+m2_c*ele2_c;
    }




    /*oneEleOneRow1_28(inputRow, numMul, outputRow, element);
    oneEleOneRow1_28(inputRow+1, numMul, outputRow, element+1);
    oneEleOneRow1_28(inputRow+2, numMul, outputRow, element+2);*/
}





void threeEleThreeRow_8(float* inputRow, int numMul, float* outputRow, float* element)
{
    //this is sliding window
    // 1. we load whole bunch
    float32x4_t ele0Dup = vld1q_dup_f32(element);
    float32x4_t ele1Dup = vld1q_dup_f32(element+1);
    float32x4_t ele2Dup = vld1q_dup_f32(element+2);
    float* curBatch = inputRow;
    float* curOutBatch = outputRow;
    int num8 = numMul/8;
    for(int i = 0; i < num8; i++ ) {

        float32x4_t input0 = vld1q_f32(curBatch);
        float32x4_t input1 = vld1q_f32(curBatch + 1);
        float32x4_t input2 = vld1q_f32(curBatch + 2);

        float32x4_t input0_1 = vld1q_f32(curBatch+4);
        float32x4_t input1_1 = vld1q_f32(curBatch + 5);
        float32x4_t input2_1 = vld1q_f32(curBatch + 6);

        float32x4_t output0 = vld1q_f32(curOutBatch);
        float32x4_t output0_1 = vld1q_f32(curOutBatch+4);

        float32x4_t result1;
        float32x4_t result1_1;

        result1 = vmlaq_f32(output0, ele0Dup, input0);
        result1_1 = vmlaq_f32(output0_1, ele0Dup, input0_1);

        result1 = vmlaq_f32(result1, ele1Dup, input1);
        result1_1 = vmlaq_f32(result1_1, ele1Dup, input1_1);

        result1 = vmlaq_f32(result1, ele2Dup, input2);
        result1_1 = vmlaq_f32(result1_1, ele2Dup, input2_1);

        vst1q_f32(curOutBatch, result1);
        vst1q_f32(curOutBatch+4, result1_1);
        curBatch+=8;
        curOutBatch+=8;
    }

    int done = num8*8;
    curBatch = inputRow+done;
    curOutBatch = outputRow+done;


    float ele0 = *element;
    float ele1 = *(element+1);
    float ele2 = *(element+2);
    for(int i = 0; i < numMul%8; i++)
    {
        float m0 = *(curBatch+i);
        float m1 = *(curBatch+i+1);
        float m2 = *(curBatch+i+2);
        *(curOutBatch+i) += m0*ele0+m1*ele1+m2*ele2;
    }




    /*oneEleOneRow1_28(inputRow, numMul, outputRow, element);
    oneEleOneRow1_28(inputRow+1, numMul, outputRow, element+1);
    oneEleOneRow1_28(inputRow+2, numMul, outputRow, element+2);*/
}

void numericcal_RowMajor_3Row_2c(float* inputImage, int imgWidth, int imgHeight, int channel,
                              float* kernels, int numKnls, int knlWidth, int knlHeight,
                              float* output, int outputWidth, int outputHeight)
{
    // only support this configuration for now
    assert(knlWidth == 3 && knlHeight ==3 && numKnls ==1);


    float* output1Row = output;
    float* input1Row = inputImage;
    //int rowMul = outputWidth;
    int offsetIn = 0;
    int offsetOut = 0;

    int knlPerChannelSize = knlWidth*numKnls;
    int imgRowSize = channel*imgWidth;
    float* curInputRow = input1Row;
    float* curKnlChannel = kernels;

    int num2c = channel/2;
    int remain = channel%2;
    int lastOffsetChannel = num2c*2*knlPerChannelSize;
    // first row
    for(int cInd = 0; cInd < num2c; cInd++) {
        int offsetChannel = cInd*2*knlPerChannelSize;
        threeEleThreeRow_2c(curInputRow, outputWidth, output1Row, curKnlChannel+offsetChannel,imgWidth,knlPerChannelSize);
        curInputRow+=2*imgWidth;
    }
    if(remain!=0) {
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+lastOffsetChannel);
        curInputRow+=imgWidth;
    }

    int knlPerRowSize = knlPerChannelSize*channel;
    for(int cInd = 0; cInd < num2c; cInd++) {
        int offsetChannel = cInd*2*knlPerChannelSize;
        threeEleThreeRow_2c(curInputRow, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ offsetChannel/*kernel[3]*/,imgWidth,knlPerChannelSize);
        threeEleThreeRow_2c(curInputRow, outputWidth, output1Row + outputWidth, curKnlChannel+offsetChannel,imgWidth,knlPerChannelSize);
        curInputRow+=2*imgWidth;
    }
    if(remain!=0) {
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ lastOffsetChannel);
        threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth, curKnlChannel+lastOffsetChannel);
        curInputRow+=imgWidth;
    }



    int steadyStateRowNum = imgHeight-2*(knlHeight-1);
    //steady state
    for(int k=0; k< steadyStateRowNum ; k++) {

        /*for(int cInd = 0; cInd < channel; cInd++) {
            int offsetChannel = cInd*knlPerChannelSize;
            threeEleThreeRow_8(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);

            threeEleThreeRow_8(curInputRow, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ offsetChannel );

            threeEleThreeRow_8(curInputRow, outputWidth, output1Row + outputWidth*2, curKnlChannel+ offsetChannel );
            curInputRow+=imgWidth;
        }*/
        for(int cInd = 0; cInd < num2c; cInd++) {
            int offsetChannel = cInd*2*knlPerChannelSize;
            threeEleThreeRow_2c(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel,imgWidth,knlPerChannelSize);

            threeEleThreeRow_2c(curInputRow, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ offsetChannel, imgWidth, knlPerChannelSize );

            threeEleThreeRow_2c(curInputRow, outputWidth, output1Row + outputWidth*2, curKnlChannel+ offsetChannel, imgWidth, knlPerChannelSize );
            curInputRow+=2*imgWidth;
        }
        if(remain!=0) {
            threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ lastOffsetChannel);

            threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ lastOffsetChannel );

            threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth*2, curKnlChannel+ lastOffsetChannel );
            curInputRow+=imgWidth;
        }

        output1Row+=(outputWidth*numKnls);
    }
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow_8(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);

        threeEleThreeRow_8(curInputRow , outputWidth, output1Row+outputWidth, curKnlChannel+ knlPerRowSize+ offsetChannel);
        curInputRow+=imgWidth;
    }
    output1Row+=(outputWidth*numKnls);

    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd * knlPerChannelSize;
        threeEleThreeRow_8(curInputRow , outputWidth, output1Row , curKnlChannel+ 2*knlPerRowSize+ offsetChannel);
        curInputRow+=imgWidth;
    }
}



// here we do three rows together
void numericcal_RowMajor_3Row(float* inputImage, int imgWidth, int imgHeight, int channel,
                         float* kernels, int numKnls, int knlWidth, int knlHeight,
                         float* output, int outputWidth, int outputHeight)
{
    // only support this configuration for now
    assert(knlWidth == 3 && knlHeight ==3 && numKnls ==1);


    float* output1Row = output;
    float* input1Row = inputImage;
    //int rowMul = outputWidth;
    int offsetIn = 0;
    int offsetOut = 0;

    int knlPerChannelSize = knlWidth*numKnls;
    int imgRowSize = channel*imgWidth;
    float* curInputRow = input1Row;
    float* curKnlChannel = kernels;
    // first row
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+offsetChannel);
        curInputRow+=imgWidth;
    }
    int knlPerRowSize = knlPerChannelSize*channel;
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ offsetChannel/*kernel[3]*/);

        threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth, curKnlChannel+offsetChannel);
        curInputRow+=imgWidth;
    }

    int steadyStateRowNum = imgHeight-2*(knlHeight-1);
    //steady state
    for(int k=0; k< steadyStateRowNum ; k++) {

        for(int cInd = 0; cInd < channel; cInd++) {
            int offsetChannel = cInd*knlPerChannelSize;
            threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);

            threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth,curKnlChannel+ knlPerRowSize+ offsetChannel );

            threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth*2, curKnlChannel+ offsetChannel );
            curInputRow+=imgWidth;
        }
        output1Row+=(outputWidth*numKnls);
    }
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);

        threeEleThreeRow(curInputRow , outputWidth, output1Row+outputWidth, curKnlChannel+ knlPerRowSize+ offsetChannel);
        curInputRow+=imgWidth;
    }
    output1Row+=(outputWidth*numKnls);

    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd * knlPerChannelSize;
        threeEleThreeRow(curInputRow , outputWidth, output1Row , curKnlChannel+ 2*knlPerRowSize+ offsetChannel);
        curInputRow+=imgWidth;
    }
}



void threeEleThreeRow_3OutMerge(float* inputRow, int numMul, float* outputRow, float* element, int knlPerRowSize)
{
    /*float* curInputRow = inputRow;
    int outputWidth = numMul;
    float* output1Row = outputRow;
    threeEleThreeRow(curInputRow, outputWidth, output1Row, element+ 2*knlPerRowSize);
    threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth,element+ knlPerRowSize );
    threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth*2, element );*/

    //this is sliding window
    // 1. we load whole bunch
    float32x4_t ele0Dup = vld1q_dup_f32(element);
    float32x4_t ele1Dup = vld1q_dup_f32(element+1);
    float32x4_t ele2Dup = vld1q_dup_f32(element+2);


    float32x4_t ele3Dup = vld1q_dup_f32(element+knlPerRowSize);
    float32x4_t ele4Dup = vld1q_dup_f32(element+knlPerRowSize+1);
    float32x4_t ele5Dup = vld1q_dup_f32(element+knlPerRowSize+2);


    float32x4_t ele6Dup = vld1q_dup_f32(element+knlPerRowSize*2);
    float32x4_t ele7Dup = vld1q_dup_f32(element+knlPerRowSize*2+1);
    float32x4_t ele8Dup = vld1q_dup_f32(element+knlPerRowSize*2+2);


    float* curBatch = inputRow;
    float* curOutBatch = outputRow;
    int num4 = numMul/4;
    for(int i = 0; i < num4; i++ ) {

        float32x4_t input0 = vld1q_f32(curBatch);
        float32x4_t input1 = vld1q_f32(curBatch + 1);
        float32x4_t input2 = vld1q_f32(curBatch + 2);

        float32x4_t output0 = vld1q_f32(curOutBatch);
        float32x4_t output1 = vld1q_f32(curOutBatch+numMul);
        float32x4_t output2 = vld1q_f32(curOutBatch+2*numMul);

        float32x4_t result1;
        result1 = vmlaq_f32(output0, ele6Dup, input0);
        result1 = vmlaq_f32(result1, ele7Dup, input1);
        result1 = vmlaq_f32(result1, ele8Dup, input2);
        vst1q_f32(curOutBatch, result1);

        float32x4_t result2;
        result2 = vmlaq_f32(output1, ele3Dup, input0);
        result2 = vmlaq_f32(result2, ele4Dup, input1);
        result2 = vmlaq_f32(result2, ele5Dup, input2);
        vst1q_f32(curOutBatch+numMul, result2);

        float32x4_t result3;
        result3 = vmlaq_f32(output2, ele0Dup, input0);
        result3 = vmlaq_f32(result3, ele1Dup, input1);
        result3 = vmlaq_f32(result3, ele2Dup, input2);
        vst1q_f32(curOutBatch+numMul*2, result3);

        curBatch+=4;
        curOutBatch+=4;
    }

    int done = num4*4;
    curBatch = inputRow+done;
    curOutBatch = outputRow+done;


    float ele0 = *element;
    float ele1 = *(element+1);
    float ele2 = *(element+2);

    float ele3 = *(element+knlPerRowSize);
    float ele4 = *(element+knlPerRowSize+1);
    float ele5 = *(element+knlPerRowSize+2);

    float ele6 = *(element+2*knlPerRowSize);
    float ele7 = *(element+2*knlPerRowSize+1);
    float ele8 = *(element+2*knlPerRowSize+2);


    for(int i = 0; i < numMul%4; i++)
    {
        float m0 = *(curBatch+i);
        float m1 = *(curBatch+i+1);
        float m2 = *(curBatch+i+2);
        *(curOutBatch+i) += m0*ele6+m1*ele7+m2*ele8;
        *(curOutBatch+numMul+i) += m0*ele3+m1*ele4+m2*ele5;
        *(curOutBatch+2*numMul+i) += m0*ele0+m1*ele1+m2*ele2;
    }




    /*oneEleOneRow1_28(inputRow, numMul, outputRow, element);
    oneEleOneRow1_28(inputRow+1, numMul, outputRow, element+1);
    oneEleOneRow1_28(inputRow+2, numMul, outputRow, element+2);*/
}

void numericcal_RowMajor_3Row_3OutMergeSteady(float* inputImage, int imgWidth, int imgHeight, int channel,
                              float* kernels, int numKnls, int knlWidth, int knlHeight,
                              float* output, int outputWidth, int outputHeight)
{
    // only support this configuration for now
    assert(knlWidth == 3 && knlHeight ==3 && numKnls ==1);


    float* output1Row = output;
    float* input1Row = inputImage;
    //int rowMul = outputWidth;
    int offsetIn = 0;
    int offsetOut = 0;

    int knlPerChannelSize = knlWidth*numKnls;
    int imgRowSize = channel*imgWidth;
    float* curInputRow = input1Row;
    float* curKnlChannel = kernels;
    // first row
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+offsetChannel);
        curInputRow+=imgWidth;
    }
    int knlPerRowSize = knlPerChannelSize*channel;
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ knlPerRowSize+ offsetChannel/*kernel[3]*/);

        threeEleThreeRow(curInputRow, outputWidth, output1Row + outputWidth, curKnlChannel+offsetChannel);
        curInputRow+=imgWidth;
    }

    int steadyStateRowNum = imgHeight-2*(knlHeight-1);
    //steady state
    for(int k=0; k< steadyStateRowNum ; k++) {

        for(int cInd = 0; cInd < channel; cInd++) {
            int offsetChannel = cInd*knlPerChannelSize;
            threeEleThreeRow_3OutMerge(curInputRow, outputWidth, output1Row, curKnlChannel+offsetChannel, knlPerRowSize);

            curInputRow+=imgWidth;
        }
        output1Row+=(outputWidth*numKnls);
    }
    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd*knlPerChannelSize;
        threeEleThreeRow(curInputRow, outputWidth, output1Row, curKnlChannel+ 2*knlPerRowSize+ offsetChannel);

        threeEleThreeRow(curInputRow , outputWidth, output1Row+outputWidth, curKnlChannel+ knlPerRowSize+ offsetChannel);
        curInputRow+=imgWidth;
    }
    output1Row+=(outputWidth*numKnls);

    for(int cInd = 0; cInd < channel; cInd++) {
        int offsetChannel = cInd * knlPerChannelSize;
        threeEleThreeRow(curInputRow , outputWidth, output1Row , curKnlChannel+ 2*knlPerRowSize+ offsetChannel);
        curInputRow+=imgWidth;
    }
}





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
        float* nextIter = curImageRow + sizePerImgRow;
        __asm__ volatile(
        "mov %0, %[nIter]\n\t"
        "PRFM PLDL2KEEP, [%0]"
        :
        : [nIter] "r" (nextIter)
        );
        numericcal_ChannelMajorCore(curImageRow, imgWidth, channel,
                                    kernels, numKnls, knlHeight, knlWidth, curOutputRow, outputWidth,
                                    0, knlHeight, 0);
        curImageRow = nextIter;
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

//














//----------------------------------------------
float goldDotProduct(float* inputPixel, float* kernelPixel, int numPoint)
{
    float result = 0.0;
    for(int k=0; k<numPoint; k++)
    {
        result += inputPixel[k]*kernelPixel[k];
    }
    return result;
}

/* gold model for row major */
/* dimension of image:
 *  height, channel, width
 * dimension of filter
 *  height,channel,numKernels,width
 * dimension of output
 *  height, channel, width
 */
void goldGetInputPointWidthMajor(float* inputImage, int imgWidth, int channel,
                             int imgWInd, int imgHInd,
                             float* singleInputAllChannel)
{
    for(int cInd = 0; cInd<channel; cInd++)
        singleInputAllChannel[cInd] = inputImage[imgHInd*imgWidth*channel+cInd*imgWidth+imgWInd];
}
void goldGetKernelPointWidthMajor(float* kernels, int knlWidth, int channel, int numKnls,
                             int knlWInd, int knlHInd, int knlInd,
                             float* singleKernelAllChannel)
{
    for(int cInd = 0; cInd < channel; cInd++)
    {
        singleKernelAllChannel[cInd] = kernels[knlHInd*channel*numKnls*knlWidth+cInd*numKnls*knlWidth+knlInd*knlWidth+knlWInd];
    }
}
float* goldGetOutputPointWidthMajor(float* output, int outputWidth, int numKnls,
                                int outWInd, int outHInd, int knlInd)
{
    return &(output[outHInd*numKnls*outputWidth+knlInd*outputWidth+outWInd]);
}


void goldWidthMajor(float* inputImage, int imgWidth, int imgHeight, int channel,
                      float* kernels, int numKnls, int knlWidth, int knlHeight,
                      float* output, int outputWidth, int outputHeight)
{
    assert(outputWidth == imgWidth-knlWidth+1);
    assert(outputHeight == imgHeight-knlHeight+1);
    float* onePointKnl = new float[channel];
    float* onePointInput = new float[channel];
    for (int outHInd = 0; outHInd < outputHeight; outHInd++)
        for (int outWInd = 0; outWInd < outputWidth; outWInd++) {
            for (int knlInd = 0; knlInd < numKnls; knlInd++) {
                float *curOutputLocation =
                        goldGetOutputPointWidthMajor(output, outputWidth, numKnls, outWInd,
                                                     outHInd, knlInd);
                for (int kHInd = 0; kHInd < knlHeight; kHInd++)
                    for (int kWInd = 0; kWInd < knlWidth; kWInd++) {


                        goldGetInputPointWidthMajor(inputImage, imgWidth, channel,
                                                    outWInd + kWInd, outHInd + kHInd,
                                                    onePointInput);

                        goldGetKernelPointWidthMajor(kernels, knlWidth, channel, numKnls,
                                                     kWInd, kHInd, knlInd, onePointKnl);
                        float curOutput = goldDotProduct(onePointInput, onePointKnl, channel);
                        *curOutputLocation += curOutput;
                    }
            }
        }
    delete [] onePointInput;
    delete [] onePointKnl;

}


/* gold model for channel major */
/* dimension of image:
 *   height, width, channel
 * dimension of kernels:
 *   height, width, numKernels, channel
 * dimension of output:
 *   height, width, channel
 * */
void goldGetInputPointChannelMajor(float* inputImage, int imgWidth, int channel,
                                 int imgWInd, int imgHInd,
                                 float* singleInputAllChannel)
{
    for(int cInd = 0; cInd<channel; cInd++)
        singleInputAllChannel[cInd] = inputImage[imgHInd*imgWidth*channel+imgWInd*channel+cInd];
}
void goldGetKernelPointChannelMajor(float* kernels, int knlWidth, int channel, int numKnls,
                                  int knlWInd, int knlHInd, int knlInd,
                                  float* singleKernelAllChannel)
{
    for(int cInd = 0; cInd < channel; cInd++)
    {
        singleKernelAllChannel[cInd] = kernels[knlHInd*channel*numKnls*knlWidth+knlWInd*numKnls*channel+knlInd*channel+cInd];
    }
}
float* goldGetOutputPointChannelMajor(float* output, int outputWidth, int numKnls,
                                    int outWInd, int outHInd, int knlInd)
{
    return &(output[outHInd*numKnls*outputWidth+outWInd*numKnls+knlInd]);
}
void goldChannelMajor(float* inputImage, int imgWidth, int imgHeight, int channel,
                      float* kernels, int numKnls, int knlWidth, int knlHeight,
                      float* output, int outputWidth, int outputHeight)
{
    assert(outputWidth == imgWidth-knlWidth+1);
    assert(outputHeight == imgHeight-knlHeight+1);
    float* onePointKnl = new float[channel];
    float* onePointInput = new float[channel];
    for (int outHInd = 0; outHInd < outputHeight; outHInd++)
        for (int outWInd = 0; outWInd < outputWidth; outWInd++) {
            for (int knlInd = 0; knlInd < numKnls; knlInd++) {
                float *curOutputLocation =
                        goldGetOutputPointChannelMajor(output, outputWidth, numKnls, outWInd,
                                                     outHInd, knlInd);
                for (int kHInd = 0; kHInd < knlHeight; kHInd++)
                    for (int kWInd = 0; kWInd < knlWidth; kWInd++) {


                        goldGetInputPointChannelMajor(inputImage, imgWidth, channel,
                                                    outWInd + kWInd, outHInd + kHInd,
                                                    onePointInput);

                        goldGetKernelPointChannelMajor(kernels, knlWidth, channel, numKnls,
                                                     kWInd, kHInd, knlInd, onePointKnl);
                        float curOutput = goldDotProduct(onePointInput, onePointKnl, channel);
                        *curOutputLocation += curOutput;
                    }
            }
        }
    delete [] onePointInput;
    delete [] onePointKnl;



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

/* gold model for row major */
/* dimension of image:
 *  height, channel, width
 * dimension of filter
 *  height,channel,numKernels,width
 * dimension of output
 *  height, channel, width
 */

void dumpArrayRowMajor(float* arr, int height, int width, int num, int channel)
{
    for(int hInd = 0; hInd < height; hInd++)
    {
        LOGD("[\t");
        for(int wInd = 0; wInd < width; wInd++)
        {
            for(int kInd = 0; kInd < num; kInd++)
            {
                if(kInd!=0)
                    LOGD(" |");
                for(int cInd = 0; cInd < channel; cInd++)
                {
                    LOGD("%f ",arr[hInd*width*num*channel+cInd*num*channel+kInd*wInd+wInd]);
                }
            }
            if(wInd!=width-1)
                LOGD(",");
        }
        LOGD("\t]\n");
    }
}

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
        dumpArrayRowMajor(goldenOutput, outputHeight, outputWidth, numKnls, channel);
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
        int channel_upper=72;
        int channel_lower=65;
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
            /*numericcal_ChannelMajor*/numericcal_RowMajor_3Row_3OutMergeSteady(inputImage, imgWidth, imgHeight, channel, kernels, numKnls,
                                    knlWidth,
                                    knlHeight, numericcalOutput, outputWidth,
                                    outputHeight);
        }
        long long timeSpent = timer_end(timecount);
        // now perform the comparison with the golden model

        repopulate(numericcalOutput, numKnls, outputWidth, outputHeight, 1, FILLZERO);

        numericcal_RowMajor_3Row_3OutMergeSteady(inputImage, imgWidth, imgHeight, channel, kernels, numKnls,
                                knlWidth,
                                knlHeight, numericcalOutput, outputWidth,
                                outputHeight);
        //dumpArrayRowMajor(numericcalOutput, outputHeight, outputWidth, numKnls, channel);
        goldWidthMajor(inputImage, imgWidth, imgHeight, channel, kernels, numKnls, knlWidth,
                         knlHeight,
                         goldenOutput, outputWidth, outputHeight);
       //LOGD("----------------------\n");
        //dumpArrayRowMajor(goldenOutput, outputHeight, outputWidth, numKnls, channel);
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
        LOGD("%d channel:\t%lld ms v.s. QSML\t%lld, L1 dist to Golden %f (normalized), %f (absolute)\n", channel, timeSpent/1000000, timeSpent2/1000000,norml1dis,l1dis);
    }
        deallocate(inputImage);
        deallocate(kernels);
        deallocate(numericcalOutput);
        deallocate(goldenOutput);

        //rtStringStream<<timeSpent/1000000<< "ms v.s. QSML"<<timeSpent2/1000000<<", L1 dist to Golden "<<norml1dis<<"(normalized),"<<l1dis<<"(absolute)\n";
    }

}
#endif
