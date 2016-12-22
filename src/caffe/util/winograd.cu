#include <algorithm>
#include <stdio.h>

#include "caffe/common.hpp" 
#include "caffe/util/winograd.hpp"
#include "caffe/util/math_functions.hpp"

#define NUM_THREADS 32
#define WINO6_TH 256

#define SSRC(i,j) src[sIdx+dataW*(i)+(j)]
#define DSRC(i) src[mIdx+gap*(i)]

namespace caffe{

template <typename Dtype> 
__global__ void padSrc_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int outH, int outW, int inputs, int batchs, int pad, float pData, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (outH * outW); 
		int yIdx = (idx % (outH * outW)) / outW - pad;
		int xIdx = idx % outW - pad;

		if(xIdx < 0 || xIdx >= dataW || yIdx < 0 || yIdx >= dataH)
			dst[idx] = pData; 
		else
			dst[idx] = src[highIdx * dataH * dataW + yIdx * dataW + xIdx]; 
	}
}

template <typename Dtype> 
__global__ void winoWeight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;

		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;


		dst[gIdx + 0] = src[kIdx + 0];
		dst[gIdx + gap] = ((src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2]) * 0.5);
		dst[gIdx + 2 * gap] = (src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2]) * 0.5;
		dst[gIdx + 3 * gap] = src[kIdx + 2];

		dst[gIdx + 4 * gap] = (src[kIdx + 0] + src[kIdx + 3] + src[kIdx + 6]) * 0.5 ;
		dst[gIdx + 5 * gap] = (src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]) * 0.25;
		dst[gIdx + 6 * gap] = (src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]) * 0.25;
		dst[gIdx + 7 * gap] = ( src[kIdx + 2] + src[kIdx + 5] + src[kIdx + 8]) * 0.5;

		dst[gIdx + 8 * gap] = ( src[kIdx + 0] - src[kIdx + 3] + src[kIdx + 6]) * 0.5;
		dst[gIdx + 9 * gap] =  (src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]) * 0.25;
		dst[gIdx + 10 * gap] = (src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]) * 0.25;
		dst[gIdx + 11 * gap] = ( src[kIdx + 2] - src[kIdx + 5] + src[kIdx + 8]) * 0.5;

		dst[gIdx + 12 * gap] = src[kIdx + 6];
		dst[gIdx + 13 * gap] = ( src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]) * 0.5;
		dst[gIdx + 14 * gap] = ( src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]) * 0.5;
		dst[gIdx + 15 * gap] = src[kIdx + 8];
/*
     if (blockIdx.x + threadIdx.x + blockIdx.y + threadIdx.y == 0) {
       printf("original_weight:");
       for (int i = 0; i < 9; i++) {
         if (i % 3 == 0) printf("\n");
         printf("%f ", src[kIdx + i]);
       }
       printf("\n\n");
       printf("weight:");
       for (int i = 0; i < 16; i++) {
         if (i % 4 == 0) printf("\n");
         printf("%f ", dst[gIdx + i * gap]);
       }
       printf("\n\n");
     }
*/
	}
}

template <typename Dtype> 
__global__ void wino4x4Weight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;

		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;


		//// -- project ---- ///
    dst[gIdx] =  src[kIdx + 0]/16;
    dst[gIdx + gap] =  (-src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2])/24;
    dst[gIdx + 2 * gap] =  (-src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2])/24;
    dst[gIdx + 3 * gap] =  src[kIdx + 0]/96 + src[kIdx + 1]/48 + src[kIdx + 2]/24;
    dst[gIdx + 4 * gap] =  src[kIdx + 0]/96 - src[kIdx + 1]/48 + src[kIdx + 2]/24;
    dst[gIdx + 5 * gap] =  src[kIdx + 2]/4;

    dst[gIdx + 6 * gap] =  -src[kIdx + 0]/24 - src[kIdx + 3]/24 - src[kIdx + 6]/24;
    dst[gIdx + 7 * gap] =  src[kIdx + 0]/36 + src[kIdx + 1]/36 + src[kIdx + 2]/36 + src[kIdx + 3]/36 + src[kIdx + 4]/36 + src[kIdx + 5]/36 + src[kIdx + 6]/36 + src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 8 * gap] =  src[kIdx + 0]/36 - src[kIdx + 1]/36 + src[kIdx + 2]/36 + src[kIdx + 3]/36 - src[kIdx + 4]/36 + src[kIdx + 5]/36 + src[kIdx + 6]/36 - src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 9 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/72 - src[kIdx + 2]/36 - src[kIdx + 3]/144 - src[kIdx + 4]/72 - src[kIdx + 5]/36 - src[kIdx + 6]/144 - src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 10 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/72 - src[kIdx + 2]/36 - src[kIdx + 3]/144 + src[kIdx + 4]/72 - src[kIdx + 5]/36 - src[kIdx + 6]/144 + src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 11 * gap] =  -src[kIdx + 2]/6 - src[kIdx + 5]/6 - src[kIdx + 8]/6;

    dst[gIdx + 12 * gap] =  -src[kIdx + 0]/24 + src[kIdx + 3]/24 - src[kIdx + 6]/24;
    dst[gIdx + 13 * gap] =  src[kIdx + 0]/36 + src[kIdx + 1]/36 + src[kIdx + 2]/36 - src[kIdx + 3]/36 - src[kIdx + 4]/36 - src[kIdx + 5]/36 + src[kIdx + 6]/36 + src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 14 * gap] =  src[kIdx + 0]/36 - src[kIdx + 1]/36 + src[kIdx + 2]/36 - src[kIdx + 3]/36 + src[kIdx + 4]/36 - src[kIdx + 5]/36 + src[kIdx + 6]/36 - src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 15 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/72 - src[kIdx + 2]/36 + src[kIdx + 3]/144 + src[kIdx + 4]/72 + src[kIdx + 5]/36 - src[kIdx + 6]/144 - src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 16 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/72 - src[kIdx + 2]/36 + src[kIdx + 3]/144 - src[kIdx + 4]/72 + src[kIdx + 5]/36 - src[kIdx + 6]/144 + src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 17 * gap] =  -src[kIdx + 2]/6 + src[kIdx + 5]/6 - src[kIdx + 8]/6;

    dst[gIdx + 18 * gap] =  src[kIdx + 0]/96 + src[kIdx + 3]/48 + src[kIdx + 6]/24;
    dst[gIdx + 19 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/144 - src[kIdx + 2]/144 - src[kIdx + 3]/72 - src[kIdx + 4]/72 - src[kIdx + 5]/72 - src[kIdx + 6]/36 - src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 20 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/144 - src[kIdx + 2]/144 - src[kIdx + 3]/72 + src[kIdx + 4]/72 - src[kIdx + 5]/72 - src[kIdx + 6]/36 + src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 21 * gap] =  src[kIdx + 0]/576 + src[kIdx + 1]/288 + src[kIdx + 2]/144 + src[kIdx + 3]/288 + src[kIdx + 4]/144 + src[kIdx + 5]/72 + src[kIdx + 6]/144 + src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 22 * gap] =  src[kIdx + 0]/576 - src[kIdx + 1]/288 + src[kIdx + 2]/144 + src[kIdx + 3]/288 - src[kIdx + 4]/144 + src[kIdx + 5]/72 + src[kIdx + 6]/144 - src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 23 * gap] =  src[kIdx + 2]/24 + src[kIdx + 5]/12 + src[kIdx + 8]/6;

    dst[gIdx + 24 * gap] =  src[kIdx + 0]/96 - src[kIdx + 3]/48 + src[kIdx + 6]/24;
    dst[gIdx + 25 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/144 - src[kIdx + 2]/144 + src[kIdx + 3]/72 + src[kIdx + 4]/72 + src[kIdx + 5]/72 - src[kIdx + 6]/36 - src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 26 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/144 - src[kIdx + 2]/144 + src[kIdx + 3]/72 - src[kIdx + 4]/72 + src[kIdx + 5]/72 - src[kIdx + 6]/36 + src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 27 * gap] =  src[kIdx + 0]/576 + src[kIdx + 1]/288 + src[kIdx + 2]/144 - src[kIdx + 3]/288 - src[kIdx + 4]/144 - src[kIdx + 5]/72 + src[kIdx + 6]/144 + src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 28 * gap] =  src[kIdx + 0]/576 - src[kIdx + 1]/288 + src[kIdx + 2]/144 - src[kIdx + 3]/288 + src[kIdx + 4]/144 - src[kIdx + 5]/72 + src[kIdx + 6]/144 - src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 29 * gap] =  src[kIdx + 2]/24 - src[kIdx + 5]/12 + src[kIdx + 8]/6;

    dst[gIdx + 30 * gap] =  src[kIdx + 6]/4;
    dst[gIdx + 31 * gap] =  -src[kIdx + 6]/6 - src[kIdx + 7]/6 - src[kIdx + 8]/6;
    dst[gIdx + 32 * gap] =  -src[kIdx + 6]/6 + src[kIdx + 7]/6 - src[kIdx + 8]/6;
    dst[gIdx + 33 * gap] =  src[kIdx + 6]/24 + src[kIdx + 7]/12 + src[kIdx + 8]/6;
    dst[gIdx + 34 * gap] =  src[kIdx + 6]/24 - src[kIdx + 7]/12 + src[kIdx + 8]/6;
    dst[gIdx + 35 * gap] =  src[kIdx + 8];

/*
    if (blockIdx.x + threadIdx.x + blockIdx.y + threadIdx.y == 0) {
      printf("original_weight:");
      for (int i = 0; i < 9; i++) {
        if (i % 3 == 0) printf("\n");
        printf("%f ", src[kIdx + i]);
      }
      printf("\n\n");
      printf("weight:");
      for (int i = 0; i < 36; i++) {
        if (i % 6 == 0) printf("\n");        
        printf("%f ", dst[gIdx + i * gap]);
      }
      printf("\n\n");
    }
    */
	}
}

template <typename Dtype> 
__global__ void wino4x4Weight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums, int zero_idx)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;

		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;


		//// -- project ---- ///


    dst[gIdx] =  src[kIdx + 0]/16;
    dst[gIdx + gap] =  (-src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2])/24;
    dst[gIdx + 2 * gap] =  (-src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2])/24;
    dst[gIdx + 3 * gap] =  src[kIdx + 0]/96 + src[kIdx + 1]/48 + src[kIdx + 2]/24;
    dst[gIdx + 4 * gap] =  src[kIdx + 0]/96 - src[kIdx + 1]/48 + src[kIdx + 2]/24;
    dst[gIdx + 5 * gap] =  src[kIdx + 2]/4;

    dst[gIdx + 6 * gap] =  -src[kIdx + 0]/24 - src[kIdx + 3]/24 - src[kIdx + 6]/24;
    dst[gIdx + 7 * gap] =  src[kIdx + 0]/36 + src[kIdx + 1]/36 + src[kIdx + 2]/36 + src[kIdx + 3]/36 + src[kIdx + 4]/36 + src[kIdx + 5]/36 + src[kIdx + 6]/36 + src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 8 * gap] =  src[kIdx + 0]/36 - src[kIdx + 1]/36 + src[kIdx + 2]/36 + src[kIdx + 3]/36 - src[kIdx + 4]/36 + src[kIdx + 5]/36 + src[kIdx + 6]/36 - src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 9 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/72 - src[kIdx + 2]/36 - src[kIdx + 3]/144 - src[kIdx + 4]/72 - src[kIdx + 5]/36 - src[kIdx + 6]/144 - src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 10 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/72 - src[kIdx + 2]/36 - src[kIdx + 3]/144 + src[kIdx + 4]/72 - src[kIdx + 5]/36 - src[kIdx + 6]/144 + src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 11 * gap] =  -src[kIdx + 2]/6 - src[kIdx + 5]/6 - src[kIdx + 8]/6;

    dst[gIdx + 12 * gap] =  -src[kIdx + 0]/24 + src[kIdx + 3]/24 - src[kIdx + 6]/24;
    dst[gIdx + 13 * gap] =  src[kIdx + 0]/36 + src[kIdx + 1]/36 + src[kIdx + 2]/36 - src[kIdx + 3]/36 - src[kIdx + 4]/36 - src[kIdx + 5]/36 + src[kIdx + 6]/36 + src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 14 * gap] =  src[kIdx + 0]/36 - src[kIdx + 1]/36 + src[kIdx + 2]/36 - src[kIdx + 3]/36 + src[kIdx + 4]/36 - src[kIdx + 5]/36 + src[kIdx + 6]/36 - src[kIdx + 7]/36 + src[kIdx + 8]/36;
    dst[gIdx + 15 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/72 - src[kIdx + 2]/36 + src[kIdx + 3]/144 + src[kIdx + 4]/72 + src[kIdx + 5]/36 - src[kIdx + 6]/144 - src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 16 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/72 - src[kIdx + 2]/36 + src[kIdx + 3]/144 - src[kIdx + 4]/72 + src[kIdx + 5]/36 - src[kIdx + 6]/144 + src[kIdx + 7]/72 - src[kIdx + 8]/36;
    dst[gIdx + 17 * gap] =  -src[kIdx + 2]/6 + src[kIdx + 5]/6 - src[kIdx + 8]/6;

    dst[gIdx + 18 * gap] =  src[kIdx + 0]/96 + src[kIdx + 3]/48 + src[kIdx + 6]/24;
    dst[gIdx + 19 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/144 - src[kIdx + 2]/144 - src[kIdx + 3]/72 - src[kIdx + 4]/72 - src[kIdx + 5]/72 - src[kIdx + 6]/36 - src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 20 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/144 - src[kIdx + 2]/144 - src[kIdx + 3]/72 + src[kIdx + 4]/72 - src[kIdx + 5]/72 - src[kIdx + 6]/36 + src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 21 * gap] =  src[kIdx + 0]/576 + src[kIdx + 1]/288 + src[kIdx + 2]/144 + src[kIdx + 3]/288 + src[kIdx + 4]/144 + src[kIdx + 5]/72 + src[kIdx + 6]/144 + src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 22 * gap] =  src[kIdx + 0]/576 - src[kIdx + 1]/288 + src[kIdx + 2]/144 + src[kIdx + 3]/288 - src[kIdx + 4]/144 + src[kIdx + 5]/72 + src[kIdx + 6]/144 - src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 23 * gap] =  src[kIdx + 2]/24 + src[kIdx + 5]/12 + src[kIdx + 8]/6;

    dst[gIdx + 24 * gap] =  src[kIdx + 0]/96 - src[kIdx + 3]/48 + src[kIdx + 6]/24;
    dst[gIdx + 25 * gap] =  -src[kIdx + 0]/144 - src[kIdx + 1]/144 - src[kIdx + 2]/144 + src[kIdx + 3]/72 + src[kIdx + 4]/72 + src[kIdx + 5]/72 - src[kIdx + 6]/36 - src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 26 * gap] =  -src[kIdx + 0]/144 + src[kIdx + 1]/144 - src[kIdx + 2]/144 + src[kIdx + 3]/72 - src[kIdx + 4]/72 + src[kIdx + 5]/72 - src[kIdx + 6]/36 + src[kIdx + 7]/36 - src[kIdx + 8]/36;
    dst[gIdx + 27 * gap] =  src[kIdx + 0]/576 + src[kIdx + 1]/288 + src[kIdx + 2]/144 - src[kIdx + 3]/288 - src[kIdx + 4]/144 - src[kIdx + 5]/72 + src[kIdx + 6]/144 + src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 28 * gap] =  src[kIdx + 0]/576 - src[kIdx + 1]/288 + src[kIdx + 2]/144 - src[kIdx + 3]/288 + src[kIdx + 4]/144 - src[kIdx + 5]/72 + src[kIdx + 6]/144 - src[kIdx + 7]/72 + src[kIdx + 8]/36;
    dst[gIdx + 29 * gap] =  src[kIdx + 2]/24 - src[kIdx + 5]/12 + src[kIdx + 8]/6;

    dst[gIdx + 30 * gap] =  src[kIdx + 6]/4;
    dst[gIdx + 31 * gap] =  -src[kIdx + 6]/6 - src[kIdx + 7]/6 - src[kIdx + 8]/6;
    dst[gIdx + 32 * gap] =  -src[kIdx + 6]/6 + src[kIdx + 7]/6 - src[kIdx + 8]/6;
    dst[gIdx + 33 * gap] =  src[kIdx + 6]/24 + src[kIdx + 7]/12 + src[kIdx + 8]/6;
    dst[gIdx + 34 * gap] =  src[kIdx + 6]/24 - src[kIdx + 7]/12 + src[kIdx + 8]/6;
    dst[gIdx + 35 * gap] =  src[kIdx + 8]; 

	}
}

template <typename Dtype> 
__global__ void wino6x6Weight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;

		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;


		//// -- project ---- ///

dst[gIdx +  0  * gap] =  src[kIdx + 0];
dst[gIdx +  1  * gap] =  -2*src[kIdx + 0]/9 - 2*src[kIdx + 1]/9 - 2*src[kIdx + 2]/9;
dst[gIdx +  2  * gap] =  -2*src[kIdx + 0]/9 + 2*src[kIdx + 1]/9 - 2*src[kIdx + 2]/9;
dst[gIdx +  3  * gap] =  src[kIdx + 0]/90 + src[kIdx + 1]/45 + 2*src[kIdx + 2]/45;
dst[gIdx +  4  * gap] =  src[kIdx + 0]/90 - src[kIdx + 1]/45 + 2*src[kIdx + 2]/45;
dst[gIdx +  5  * gap] =  32*src[kIdx + 0]/45 + 16*src[kIdx + 1]/45 + 8*src[kIdx + 2]/45;
dst[gIdx +  6  * gap] =  32*src[kIdx + 0]/45 - 16*src[kIdx + 1]/45 + 8*src[kIdx + 2]/45;
dst[gIdx +  7  * gap] =  src[kIdx + 2];
;
dst[gIdx +  8  * gap] =  -2*src[kIdx + 0]/9 - 2*src[kIdx + 3]/9 - 2*src[kIdx + 6]/9;
dst[gIdx +  9  * gap] =  4*src[kIdx + 0]/81 + 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 + 4*src[kIdx + 3]/81 + 4*src[kIdx + 4]/81 + 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 + 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  10  * gap] =  4*src[kIdx + 0]/81 - 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 + 4*src[kIdx + 3]/81 - 4*src[kIdx + 4]/81 + 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 - 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  11  * gap] =  -src[kIdx + 0]/405 - 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 - src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 - 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 - 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  12  * gap] =  -src[kIdx + 0]/405 + 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 - src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 - 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 + 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  13  * gap] =  -64*src[kIdx + 0]/405 - 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 - 64*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 - 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 - 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  14  * gap] =  -64*src[kIdx + 0]/405 + 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 - 64*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 - 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 + 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  15  * gap] =  -2*src[kIdx + 2]/9 - 2*src[kIdx + 5]/9 - 2*src[kIdx + 8]/9;
;
dst[gIdx +  16  * gap] =  -2*src[kIdx + 0]/9 + 2*src[kIdx + 3]/9 - 2*src[kIdx + 6]/9;
dst[gIdx +  17  * gap] =  4*src[kIdx + 0]/81 + 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 - 4*src[kIdx + 3]/81 - 4*src[kIdx + 4]/81 - 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 + 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  18  * gap] =  4*src[kIdx + 0]/81 - 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 - 4*src[kIdx + 3]/81 + 4*src[kIdx + 4]/81 - 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 - 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  19  * gap] =  -src[kIdx + 0]/405 - 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 + src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 + 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 - 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  20  * gap] =  -src[kIdx + 0]/405 + 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 + src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 + 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 + 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  21  * gap] =  -64*src[kIdx + 0]/405 - 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 + 64*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 + 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 - 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  22  * gap] =  -64*src[kIdx + 0]/405 + 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 + 64*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 + 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 + 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  23  * gap] =  -2*src[kIdx + 2]/9 + 2*src[kIdx + 5]/9 - 2*src[kIdx + 8]/9;
;
dst[gIdx +  24  * gap] =  src[kIdx + 0]/90 + src[kIdx + 3]/45 + 2*src[kIdx + 6]/45;
dst[gIdx +  25  * gap] =  -src[kIdx + 0]/405 - src[kIdx + 1]/405 - src[kIdx + 2]/405 - 2*src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 - 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 - 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  26  * gap] =  -src[kIdx + 0]/405 + src[kIdx + 1]/405 - src[kIdx + 2]/405 - 2*src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 - 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 + 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  27  * gap] =  src[kIdx + 0]/8100 + src[kIdx + 1]/4050 + src[kIdx + 2]/2025 + src[kIdx + 3]/4050 + src[kIdx + 4]/2025 + 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 + 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  28  * gap] =  src[kIdx + 0]/8100 - src[kIdx + 1]/4050 + src[kIdx + 2]/2025 + src[kIdx + 3]/4050 - src[kIdx + 4]/2025 + 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 - 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  29  * gap] =  16*src[kIdx + 0]/2025 + 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 + 32*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 + 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 + 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  30  * gap] =  16*src[kIdx + 0]/2025 - 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 + 32*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 + 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 - 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  31  * gap] =  src[kIdx + 2]/90 + src[kIdx + 5]/45 + 2*src[kIdx + 8]/45;
;
dst[gIdx +  32  * gap] =  src[kIdx + 0]/90 - src[kIdx + 3]/45 + 2*src[kIdx + 6]/45;
dst[gIdx +  33  * gap] =  -src[kIdx + 0]/405 - src[kIdx + 1]/405 - src[kIdx + 2]/405 + 2*src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 + 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 - 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  34  * gap] =  -src[kIdx + 0]/405 + src[kIdx + 1]/405 - src[kIdx + 2]/405 + 2*src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 + 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 + 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  35  * gap] =  src[kIdx + 0]/8100 + src[kIdx + 1]/4050 + src[kIdx + 2]/2025 - src[kIdx + 3]/4050 - src[kIdx + 4]/2025 - 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 + 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  36  * gap] =  src[kIdx + 0]/8100 - src[kIdx + 1]/4050 + src[kIdx + 2]/2025 - src[kIdx + 3]/4050 + src[kIdx + 4]/2025 - 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 - 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  37  * gap] =  16*src[kIdx + 0]/2025 + 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 - 32*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 - 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 + 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  38  * gap] =  16*src[kIdx + 0]/2025 - 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 - 32*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 - 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 - 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  39  * gap] =  src[kIdx + 2]/90 - src[kIdx + 5]/45 + 2*src[kIdx + 8]/45;
;
dst[gIdx +  40  * gap] =  32*src[kIdx + 0]/45 + 16*src[kIdx + 3]/45 + 8*src[kIdx + 6]/45;
dst[gIdx +  41  * gap] =  -64*src[kIdx + 0]/405 - 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 - 32*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 - 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 - 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  42  * gap] =  -64*src[kIdx + 0]/405 + 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 - 32*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 - 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 + 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  43  * gap] =  16*src[kIdx + 0]/2025 + 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 + 8*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 + 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 + 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  44  * gap] =  16*src[kIdx + 0]/2025 - 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 + 8*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 + 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 - 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  45  * gap] =  1024*src[kIdx + 0]/2025 + 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 + 512*src[kIdx + 3]/2025 + 256*src[kIdx + 4]/2025 + 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 + 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  46  * gap] =  1024*src[kIdx + 0]/2025 - 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 + 512*src[kIdx + 3]/2025 - 256*src[kIdx + 4]/2025 + 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 - 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  47  * gap] =  32*src[kIdx + 2]/45 + 16*src[kIdx + 5]/45 + 8*src[kIdx + 8]/45;
;
dst[gIdx +  48  * gap] =  32*src[kIdx + 0]/45 - 16*src[kIdx + 3]/45 + 8*src[kIdx + 6]/45;
dst[gIdx +  49  * gap] =  -64*src[kIdx + 0]/405 - 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 + 32*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 + 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 - 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  50  * gap] =  -64*src[kIdx + 0]/405 + 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 + 32*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 + 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 + 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  51  * gap] =  16*src[kIdx + 0]/2025 + 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 - 8*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 - 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 + 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  52  * gap] =  16*src[kIdx + 0]/2025 - 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 - 8*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 - 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 - 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  53  * gap] =  1024*src[kIdx + 0]/2025 + 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 - 512*src[kIdx + 3]/2025 - 256*src[kIdx + 4]/2025 - 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 + 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  54  * gap] =  1024*src[kIdx + 0]/2025 - 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 - 512*src[kIdx + 3]/2025 + 256*src[kIdx + 4]/2025 - 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 - 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  55  * gap] =  32*src[kIdx + 2]/45 - 16*src[kIdx + 5]/45 + 8*src[kIdx + 8]/45;
;
dst[gIdx +  56  * gap] =  src[kIdx + 6];
dst[gIdx +  57  * gap] =  -2*src[kIdx + 6]/9 - 2*src[kIdx + 7]/9 - 2*src[kIdx + 8]/9;
dst[gIdx +  58  * gap] =  -2*src[kIdx + 6]/9 + 2*src[kIdx + 7]/9 - 2*src[kIdx + 8]/9;
dst[gIdx +  59  * gap] =  src[kIdx + 6]/90 + src[kIdx + 7]/45 + 2*src[kIdx + 8]/45;
dst[gIdx +  60  * gap] =  src[kIdx + 6]/90 - src[kIdx + 7]/45 + 2*src[kIdx + 8]/45;
dst[gIdx +  61  * gap] =  32*src[kIdx + 6]/45 + 16*src[kIdx + 7]/45 + 8*src[kIdx + 8]/45;
dst[gIdx +  62  * gap] =  32*src[kIdx + 6]/45 - 16*src[kIdx + 7]/45 + 8*src[kIdx + 8]/45;
dst[gIdx +  63  * gap] =  src[kIdx + 8];




	}
}

template <typename Dtype> 
__global__ void wino6x6Weight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums, int zero_idx)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;

		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;


		//// -- project ---- ///

dst[gIdx +  0  * gap] =  src[kIdx + 0];
dst[gIdx +  1  * gap] =  -2*src[kIdx + 0]/9 - 2*src[kIdx + 1]/9 - 2*src[kIdx + 2]/9;
dst[gIdx +  2  * gap] =  -2*src[kIdx + 0]/9 + 2*src[kIdx + 1]/9 - 2*src[kIdx + 2]/9;
dst[gIdx +  3  * gap] =  src[kIdx + 0]/90 + src[kIdx + 1]/45 + 2*src[kIdx + 2]/45;
dst[gIdx +  4  * gap] =  src[kIdx + 0]/90 - src[kIdx + 1]/45 + 2*src[kIdx + 2]/45;
dst[gIdx +  5  * gap] =  32*src[kIdx + 0]/45 + 16*src[kIdx + 1]/45 + 8*src[kIdx + 2]/45;
dst[gIdx +  6  * gap] =  32*src[kIdx + 0]/45 - 16*src[kIdx + 1]/45 + 8*src[kIdx + 2]/45;
dst[gIdx +  7  * gap] =  src[kIdx + 2];
;
dst[gIdx +  8  * gap] =  -2*src[kIdx + 0]/9 - 2*src[kIdx + 3]/9 - 2*src[kIdx + 6]/9;
dst[gIdx +  9  * gap] =  4*src[kIdx + 0]/81 + 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 + 4*src[kIdx + 3]/81 + 4*src[kIdx + 4]/81 + 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 + 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  10  * gap] =  4*src[kIdx + 0]/81 - 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 + 4*src[kIdx + 3]/81 - 4*src[kIdx + 4]/81 + 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 - 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  11  * gap] =  -src[kIdx + 0]/405 - 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 - src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 - 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 - 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  12  * gap] =  -src[kIdx + 0]/405 + 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 - src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 - 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 + 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  13  * gap] =  -64*src[kIdx + 0]/405 - 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 - 64*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 - 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 - 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  14  * gap] =  -64*src[kIdx + 0]/405 + 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 - 64*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 - 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 + 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  15  * gap] =  -2*src[kIdx + 2]/9 - 2*src[kIdx + 5]/9 - 2*src[kIdx + 8]/9;
;
dst[gIdx +  16  * gap] =  -2*src[kIdx + 0]/9 + 2*src[kIdx + 3]/9 - 2*src[kIdx + 6]/9;
dst[gIdx +  17  * gap] =  4*src[kIdx + 0]/81 + 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 - 4*src[kIdx + 3]/81 - 4*src[kIdx + 4]/81 - 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 + 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  18  * gap] =  4*src[kIdx + 0]/81 - 4*src[kIdx + 1]/81 + 4*src[kIdx + 2]/81 - 4*src[kIdx + 3]/81 + 4*src[kIdx + 4]/81 - 4*src[kIdx + 5]/81 + 4*src[kIdx + 6]/81 - 4*src[kIdx + 7]/81 + 4*src[kIdx + 8]/81;
dst[gIdx +  19  * gap] =  -src[kIdx + 0]/405 - 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 + src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 + 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 - 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  20  * gap] =  -src[kIdx + 0]/405 + 2*src[kIdx + 1]/405 - 4*src[kIdx + 2]/405 + src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 + 4*src[kIdx + 5]/405 - src[kIdx + 6]/405 + 2*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  21  * gap] =  -64*src[kIdx + 0]/405 - 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 + 64*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 + 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 - 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  22  * gap] =  -64*src[kIdx + 0]/405 + 32*src[kIdx + 1]/405 - 16*src[kIdx + 2]/405 + 64*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 + 16*src[kIdx + 5]/405 - 64*src[kIdx + 6]/405 + 32*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  23  * gap] =  -2*src[kIdx + 2]/9 + 2*src[kIdx + 5]/9 - 2*src[kIdx + 8]/9;
;
dst[gIdx +  24  * gap] =  src[kIdx + 0]/90 + src[kIdx + 3]/45 + 2*src[kIdx + 6]/45;
dst[gIdx +  25  * gap] =  -src[kIdx + 0]/405 - src[kIdx + 1]/405 - src[kIdx + 2]/405 - 2*src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 - 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 - 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  26  * gap] =  -src[kIdx + 0]/405 + src[kIdx + 1]/405 - src[kIdx + 2]/405 - 2*src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 - 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 + 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  27  * gap] =  src[kIdx + 0]/8100 + src[kIdx + 1]/4050 + src[kIdx + 2]/2025 + src[kIdx + 3]/4050 + src[kIdx + 4]/2025 + 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 + 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  28  * gap] =  src[kIdx + 0]/8100 - src[kIdx + 1]/4050 + src[kIdx + 2]/2025 + src[kIdx + 3]/4050 - src[kIdx + 4]/2025 + 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 - 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  29  * gap] =  16*src[kIdx + 0]/2025 + 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 + 32*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 + 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 + 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  30  * gap] =  16*src[kIdx + 0]/2025 - 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 + 32*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 + 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 - 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  31  * gap] =  src[kIdx + 2]/90 + src[kIdx + 5]/45 + 2*src[kIdx + 8]/45;
;
dst[gIdx +  32  * gap] =  src[kIdx + 0]/90 - src[kIdx + 3]/45 + 2*src[kIdx + 6]/45;
dst[gIdx +  33  * gap] =  -src[kIdx + 0]/405 - src[kIdx + 1]/405 - src[kIdx + 2]/405 + 2*src[kIdx + 3]/405 + 2*src[kIdx + 4]/405 + 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 - 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  34  * gap] =  -src[kIdx + 0]/405 + src[kIdx + 1]/405 - src[kIdx + 2]/405 + 2*src[kIdx + 3]/405 - 2*src[kIdx + 4]/405 + 2*src[kIdx + 5]/405 - 4*src[kIdx + 6]/405 + 4*src[kIdx + 7]/405 - 4*src[kIdx + 8]/405;
dst[gIdx +  35  * gap] =  src[kIdx + 0]/8100 + src[kIdx + 1]/4050 + src[kIdx + 2]/2025 - src[kIdx + 3]/4050 - src[kIdx + 4]/2025 - 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 + 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  36  * gap] =  src[kIdx + 0]/8100 - src[kIdx + 1]/4050 + src[kIdx + 2]/2025 - src[kIdx + 3]/4050 + src[kIdx + 4]/2025 - 2*src[kIdx + 5]/2025 + src[kIdx + 6]/2025 - 2*src[kIdx + 7]/2025 + 4*src[kIdx + 8]/2025;
dst[gIdx +  37  * gap] =  16*src[kIdx + 0]/2025 + 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 - 32*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 - 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 + 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  38  * gap] =  16*src[kIdx + 0]/2025 - 8*src[kIdx + 1]/2025 + 4*src[kIdx + 2]/2025 - 32*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 - 8*src[kIdx + 5]/2025 + 64*src[kIdx + 6]/2025 - 32*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  39  * gap] =  src[kIdx + 2]/90 - src[kIdx + 5]/45 + 2*src[kIdx + 8]/45;
;
dst[gIdx +  40  * gap] =  32*src[kIdx + 0]/45 + 16*src[kIdx + 3]/45 + 8*src[kIdx + 6]/45;
dst[gIdx +  41  * gap] =  -64*src[kIdx + 0]/405 - 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 - 32*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 - 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 - 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  42  * gap] =  -64*src[kIdx + 0]/405 + 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 - 32*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 - 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 + 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  43  * gap] =  16*src[kIdx + 0]/2025 + 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 + 8*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 + 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 + 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  44  * gap] =  16*src[kIdx + 0]/2025 - 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 + 8*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 + 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 - 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  45  * gap] =  1024*src[kIdx + 0]/2025 + 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 + 512*src[kIdx + 3]/2025 + 256*src[kIdx + 4]/2025 + 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 + 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  46  * gap] =  1024*src[kIdx + 0]/2025 - 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 + 512*src[kIdx + 3]/2025 - 256*src[kIdx + 4]/2025 + 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 - 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  47  * gap] =  32*src[kIdx + 2]/45 + 16*src[kIdx + 5]/45 + 8*src[kIdx + 8]/45;
;
dst[gIdx +  48  * gap] =  32*src[kIdx + 0]/45 - 16*src[kIdx + 3]/45 + 8*src[kIdx + 6]/45;
dst[gIdx +  49  * gap] =  -64*src[kIdx + 0]/405 - 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 + 32*src[kIdx + 3]/405 + 32*src[kIdx + 4]/405 + 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 - 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  50  * gap] =  -64*src[kIdx + 0]/405 + 64*src[kIdx + 1]/405 - 64*src[kIdx + 2]/405 + 32*src[kIdx + 3]/405 - 32*src[kIdx + 4]/405 + 32*src[kIdx + 5]/405 - 16*src[kIdx + 6]/405 + 16*src[kIdx + 7]/405 - 16*src[kIdx + 8]/405;
dst[gIdx +  51  * gap] =  16*src[kIdx + 0]/2025 + 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 - 8*src[kIdx + 3]/2025 - 16*src[kIdx + 4]/2025 - 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 + 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  52  * gap] =  16*src[kIdx + 0]/2025 - 32*src[kIdx + 1]/2025 + 64*src[kIdx + 2]/2025 - 8*src[kIdx + 3]/2025 + 16*src[kIdx + 4]/2025 - 32*src[kIdx + 5]/2025 + 4*src[kIdx + 6]/2025 - 8*src[kIdx + 7]/2025 + 16*src[kIdx + 8]/2025;
dst[gIdx +  53  * gap] =  1024*src[kIdx + 0]/2025 + 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 - 512*src[kIdx + 3]/2025 - 256*src[kIdx + 4]/2025 - 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 + 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  54  * gap] =  1024*src[kIdx + 0]/2025 - 512*src[kIdx + 1]/2025 + 256*src[kIdx + 2]/2025 - 512*src[kIdx + 3]/2025 + 256*src[kIdx + 4]/2025 - 128*src[kIdx + 5]/2025 + 256*src[kIdx + 6]/2025 - 128*src[kIdx + 7]/2025 + 64*src[kIdx + 8]/2025;
dst[gIdx +  55  * gap] =  32*src[kIdx + 2]/45 - 16*src[kIdx + 5]/45 + 8*src[kIdx + 8]/45;
;
dst[gIdx +  56  * gap] =  src[kIdx + 6];
dst[gIdx +  57  * gap] =  -2*src[kIdx + 6]/9 - 2*src[kIdx + 7]/9 - 2*src[kIdx + 8]/9;
dst[gIdx +  58  * gap] =  -2*src[kIdx + 6]/9 + 2*src[kIdx + 7]/9 - 2*src[kIdx + 8]/9;
dst[gIdx +  59  * gap] =  src[kIdx + 6]/90 + src[kIdx + 7]/45 + 2*src[kIdx + 8]/45;
dst[gIdx +  60  * gap] =  src[kIdx + 6]/90 - src[kIdx + 7]/45 + 2*src[kIdx + 8]/45;
dst[gIdx +  61  * gap] =  32*src[kIdx + 6]/45 + 16*src[kIdx + 7]/45 + 8*src[kIdx + 8]/45;
dst[gIdx +  62  * gap] =  32*src[kIdx + 6]/45 - 16*src[kIdx + 7]/45 + 8*src[kIdx + 8]/45;
dst[gIdx +  63  * gap] =  src[kIdx + 8];





	}
}

template <typename Dtype> 
__global__ void winoSrc_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = tileH * tileW * inputs * batchs;
		int batchIdx = idx / (tileH * tileW * inputs);
		int inputIdx = (idx / (tileH * tileW)) % inputs ;
		int yIdx = (idx % (tileH * tileW)) / tileW ;
		int xIdx = idx % tileW;
		int bIdx =  idx % gap;
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;

		dst[bIdx + 0] = src[sIdx + 0]  - src[sIdx + 2] - src[sIdx + 2 * dataW] + src[sIdx + 2 * dataW + 2];
		dst[bIdx + gap] = src[sIdx + 1]  + src[sIdx + 2] - src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 2];
		dst[bIdx + 2 * gap] = -1 * src[sIdx + 1] + src[sIdx + 2] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 2];
		dst[bIdx + 3 * gap] = src[sIdx + 1] - src[sIdx + 3] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3];

		dst[bIdx + 4 * gap] = src[sIdx + dataW] - src[sIdx + dataW + 2] + src[sIdx + 2 * dataW] - src[sIdx + 2 * dataW + 2];
		dst[bIdx + 5 * gap] = src[sIdx + dataW + 1] + src[sIdx + dataW + 2] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2];
		dst[bIdx + 6 * gap] = -1 * src[sIdx + dataW + 1] + src[sIdx + dataW + 2] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2];
		dst[bIdx + 7 * gap] = src[sIdx + dataW + 1] - src[sIdx + dataW + 3] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3];

		dst[bIdx + 8 * gap] = -1 * src[sIdx + dataW] + src[sIdx + dataW + 2] + src[sIdx + 2 * dataW] - src[sIdx + 2 * dataW + 2];
		dst[bIdx + 9 * gap]  = -1 * src[sIdx + dataW + 1] - src[sIdx + dataW + 2] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2];
		dst[bIdx + 10 * gap] = src[sIdx + dataW + 1] - src[sIdx + dataW + 2] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2];
		dst[bIdx + 11 * gap] = -1 * src[sIdx + dataW + 1] + src[sIdx + dataW + 3] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3];

		dst[bIdx + 12 * gap] = src[sIdx + dataW] - src[sIdx + dataW + 2] - src[sIdx + 3 * dataW] + src[sIdx + 3 * dataW + 2];
		dst[bIdx + 13 * gap] = src[sIdx + dataW + 1] + src[sIdx + dataW + 2] - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2];
		dst[bIdx + 14 * gap] = -1 * src[sIdx + dataW + 1] + src[sIdx + dataW + 2] + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2];
		dst[bIdx + 15 * gap] = src[sIdx + dataW + 1] - src[sIdx + dataW + 3] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3];
/*
    if (threadIdx.x + blockIdx.x + threadIdx.y + blockIdx.y == 0) {
       printf("original src:\n");
       for (int i = 0; i < 4; i++) {
         for (int j = 0; j < 4; j++) {
           printf("%f ", src[sIdx + j + i * dataW]);
         }
         printf("\n");
       }
 
 
       printf("Src:");
       for (int i = 0; i < 16;i++) {
         if (i%4 == 0) printf("\n");
         printf("%f ", dst[bIdx + i * gap]);
       }
       printf("\n\n");
     }
*/
	}
}


template <typename Dtype> 
__global__ void wino4x4Src_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = tileH * tileW * inputs * batchs;
		int batchIdx = idx / (tileH * tileW * inputs);
		int inputIdx = (idx / (tileH * tileW)) % inputs ;
		int yIdx = (idx % (tileH * tileW)) / tileW ;
		int xIdx = idx % tileW;
		int bIdx =  idx % gap;
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 4 + xIdx * 4;

    //// -- project ---- ///
    dst[bIdx + 0 * gap] =  16*src[sIdx] - 20*src[sIdx + 2] + 4*src[sIdx + 4] - 20*src[sIdx + 2 * dataW] + 25*src[sIdx + 2 + 2 * dataW] - 5*src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 4 * dataW] - 5*src[sIdx + 2 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 1 * gap] =  -16*src[sIdx + 1] - 16*src[sIdx + 2] + 4*src[sIdx + 3] + 4*src[sIdx + 4] + 20*src[sIdx + 1 + 2 * dataW] + 20*src[sIdx + 2 + 2 * dataW] - 5*src[sIdx + 3 + 2 * dataW] - 5*src[sIdx + 4 + 2 * dataW] - 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] + src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 2 * gap] =  16*src[sIdx + 1] - 16*src[sIdx + 2] - 4*src[sIdx + 3] + 4*src[sIdx + 4] - 20*src[sIdx + 1 + 2 * dataW] + 20*src[sIdx + 2 + 2 * dataW] + 5*src[sIdx + 3 + 2 * dataW] - 5*src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] - src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 3 * gap] =  -8*src[sIdx + 1] - 4*src[sIdx + 2] + 8*src[sIdx + 3] + 4*src[sIdx + 4] + 10*src[sIdx + 1 + 2 * dataW] + 5*src[sIdx + 2 + 2 * dataW] - 10*src[sIdx + 3 + 2 * dataW] - 5*src[sIdx + 4 + 2 * dataW] - 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] + 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 4 * gap] =  8*src[sIdx + 1] - 4*src[sIdx + 2] - 8*src[sIdx + 3] + 4*src[sIdx + 4] - 10*src[sIdx + 1 + 2 * dataW] + 5*src[sIdx + 2 + 2 * dataW] + 10*src[sIdx + 3 + 2 * dataW] - 5*src[sIdx + 4 + 2 * dataW] + 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] - 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 5 * gap] =  16*src[sIdx + 1] - 20*src[sIdx + 3] + 4*src[sIdx + 5] - 20*src[sIdx + 1 + 2 * dataW] + 25*src[sIdx + 3 + 2 * dataW] - 5*src[sIdx + 5 + 2 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 5*src[sIdx + 3 + 4 * dataW] + src[sIdx + 5 + 4 * dataW];

    dst[bIdx + 6 * gap] =  -16*src[sIdx + dataW] + 20*src[sIdx + 2 + dataW] - 4*src[sIdx + 4 + dataW] - 16*src[sIdx + 2 * dataW] + 20*src[sIdx + 2 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 3 * dataW] - 5*src[sIdx + 2 + 3 * dataW] + src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 4 * dataW] - 5*src[sIdx + 2 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 7 * gap] =  16*src[sIdx + 1 + dataW] + 16*src[sIdx + 2 + dataW] - 4*src[sIdx + 3 + dataW] - 4*src[sIdx + 4 + dataW] + 16*src[sIdx + 1 + 2 * dataW] + 16*src[sIdx + 2 + 2 * dataW] - 4*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] - 4*src[sIdx + 1 + 3 * dataW] - 4*src[sIdx + 2 + 3 * dataW] + src[sIdx + 3 + 3 * dataW] + src[sIdx + 4 + 3 * dataW] - 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] + src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 8 * gap] =  -16*src[sIdx + 1 + dataW] + 16*src[sIdx + 2 + dataW] + 4*src[sIdx + 3 + dataW] - 4*src[sIdx + 4 + dataW] - 16*src[sIdx + 1 + 2 * dataW] + 16*src[sIdx + 2 + 2 * dataW] + 4*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 1 + 3 * dataW] - 4*src[sIdx + 2 + 3 * dataW] - src[sIdx + 3 + 3 * dataW] + src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] - src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 9 * gap] =  8*src[sIdx + 1 + dataW] + 4*src[sIdx + 2 + dataW] - 8*src[sIdx + 3 + dataW] - 4*src[sIdx + 4 + dataW] + 8*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] - 8*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] - 2*src[sIdx + 1 + 3 * dataW] - src[sIdx + 2 + 3 * dataW] + 2*src[sIdx + 3 + 3 * dataW] + src[sIdx + 4 + 3 * dataW] - 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] + 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 10 * gap] =  -8*src[sIdx + 1 + dataW] + 4*src[sIdx + 2 + dataW] + 8*src[sIdx + 3 + dataW] - 4*src[sIdx + 4 + dataW] - 8*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] + 8*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] + 2*src[sIdx + 1 + 3 * dataW] - src[sIdx + 2 + 3 * dataW] - 2*src[sIdx + 3 + 3 * dataW] + src[sIdx + 4 + 3 * dataW] + 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] - 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 11 * gap] =  -16*src[sIdx + 1 + dataW] + 20*src[sIdx + 3 + dataW] - 4*src[sIdx + 5 + dataW] - 16*src[sIdx + 1 + 2 * dataW] + 20*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 5 + 2 * dataW] + 4*src[sIdx + 1 + 3 * dataW] - 5*src[sIdx + 3 + 3 * dataW] + src[sIdx + 5 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 5*src[sIdx + 3 + 4 * dataW] + src[sIdx + 5 + 4 * dataW];

    dst[bIdx + 12 * gap] =  16*src[sIdx + dataW] - 20*src[sIdx + 2 + dataW] + 4*src[sIdx + 4 + dataW] - 16*src[sIdx + 2 * dataW] + 20*src[sIdx + 2 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] - 4*src[sIdx + 3 * dataW] + 5*src[sIdx + 2 + 3 * dataW] - src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 4 * dataW] - 5*src[sIdx + 2 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 13 * gap] =  -16*src[sIdx + 1 + dataW] - 16*src[sIdx + 2 + dataW] + 4*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] + 16*src[sIdx + 1 + 2 * dataW] + 16*src[sIdx + 2 + 2 * dataW] - 4*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 1 + 3 * dataW] + 4*src[sIdx + 2 + 3 * dataW] - src[sIdx + 3 + 3 * dataW] - src[sIdx + 4 + 3 * dataW] - 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] + src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 14 * gap] =  16*src[sIdx + 1 + dataW] - 16*src[sIdx + 2 + dataW] - 4*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] - 16*src[sIdx + 1 + 2 * dataW] + 16*src[sIdx + 2 + 2 * dataW] + 4*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] - 4*src[sIdx + 1 + 3 * dataW] + 4*src[sIdx + 2 + 3 * dataW] + src[sIdx + 3 + 3 * dataW] - src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] - src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 15 * gap] =  -8*src[sIdx + 1 + dataW] - 4*src[sIdx + 2 + dataW] + 8*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] + 8*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] - 8*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] + 2*src[sIdx + 1 + 3 * dataW] + src[sIdx + 2 + 3 * dataW] - 2*src[sIdx + 3 + 3 * dataW] - src[sIdx + 4 + 3 * dataW] - 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] + 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 16 * gap] =  8*src[sIdx + 1 + dataW] - 4*src[sIdx + 2 + dataW] - 8*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] - 8*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] + 8*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 4 + 2 * dataW] - 2*src[sIdx + 1 + 3 * dataW] + src[sIdx + 2 + 3 * dataW] + 2*src[sIdx + 3 + 3 * dataW] - src[sIdx + 4 + 3 * dataW] + 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] - 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 17 * gap] =  16*src[sIdx + 1 + dataW] - 20*src[sIdx + 3 + dataW] + 4*src[sIdx + 5 + dataW] - 16*src[sIdx + 1 + 2 * dataW] + 20*src[sIdx + 3 + 2 * dataW] - 4*src[sIdx + 5 + 2 * dataW] - 4*src[sIdx + 1 + 3 * dataW] + 5*src[sIdx + 3 + 3 * dataW] - src[sIdx + 5 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 5*src[sIdx + 3 + 4 * dataW] + src[sIdx + 5 + 4 * dataW];

    dst[bIdx + 18 * gap] =  -8*src[sIdx + dataW] + 10*src[sIdx + 2 + dataW] - 2*src[sIdx + 4 + dataW] - 4*src[sIdx + 2 * dataW] + 5*src[sIdx + 2 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] + 8*src[sIdx + 3 * dataW] - 10*src[sIdx + 2 + 3 * dataW] + 2*src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 4 * dataW] - 5*src[sIdx + 2 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 19 * gap] =  8*src[sIdx + 1 + dataW] + 8*src[sIdx + 2 + dataW] - 2*src[sIdx + 3 + dataW] - 2*src[sIdx + 4 + dataW] + 4*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] - src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] - 8*src[sIdx + 1 + 3 * dataW] - 8*src[sIdx + 2 + 3 * dataW] + 2*src[sIdx + 3 + 3 * dataW] + 2*src[sIdx + 4 + 3 * dataW] - 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] + src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 20 * gap] =  -8*src[sIdx + 1 + dataW] + 8*src[sIdx + 2 + dataW] + 2*src[sIdx + 3 + dataW] - 2*src[sIdx + 4 + dataW] - 4*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] + src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] + 8*src[sIdx + 1 + 3 * dataW] - 8*src[sIdx + 2 + 3 * dataW] - 2*src[sIdx + 3 + 3 * dataW] + 2*src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] - src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 21 * gap] =  4*src[sIdx + 1 + dataW] + 2*src[sIdx + 2 + dataW] - 4*src[sIdx + 3 + dataW] - 2*src[sIdx + 4 + dataW] + 2*src[sIdx + 1 + 2 * dataW] + src[sIdx + 2 + 2 * dataW] - 2*src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] - 4*src[sIdx + 1 + 3 * dataW] - 2*src[sIdx + 2 + 3 * dataW] + 4*src[sIdx + 3 + 3 * dataW] + 2*src[sIdx + 4 + 3 * dataW] - 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] + 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 22 * gap] =  -4*src[sIdx + 1 + dataW] + 2*src[sIdx + 2 + dataW] + 4*src[sIdx + 3 + dataW] - 2*src[sIdx + 4 + dataW] - 2*src[sIdx + 1 + 2 * dataW] + src[sIdx + 2 + 2 * dataW] + 2*src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 1 + 3 * dataW] - 2*src[sIdx + 2 + 3 * dataW] - 4*src[sIdx + 3 + 3 * dataW] + 2*src[sIdx + 4 + 3 * dataW] + 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] - 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 23 * gap] =  -8*src[sIdx + 1 + dataW] + 10*src[sIdx + 3 + dataW] - 2*src[sIdx + 5 + dataW] - 4*src[sIdx + 1 + 2 * dataW] + 5*src[sIdx + 3 + 2 * dataW] - src[sIdx + 5 + 2 * dataW] + 8*src[sIdx + 1 + 3 * dataW] - 10*src[sIdx + 3 + 3 * dataW] + 2*src[sIdx + 5 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 5*src[sIdx + 3 + 4 * dataW] + src[sIdx + 5 + 4 * dataW];

    dst[bIdx + 24 * gap] =  8*src[sIdx + dataW] - 10*src[sIdx + 2 + dataW] + 2*src[sIdx + 4 + dataW] - 4*src[sIdx + 2 * dataW] + 5*src[sIdx + 2 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] - 8*src[sIdx + 3 * dataW] + 10*src[sIdx + 2 + 3 * dataW] - 2*src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 4 * dataW] - 5*src[sIdx + 2 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 25 * gap] =  -8*src[sIdx + 1 + dataW] - 8*src[sIdx + 2 + dataW] + 2*src[sIdx + 3 + dataW] + 2*src[sIdx + 4 + dataW] + 4*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] - src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] + 8*src[sIdx + 1 + 3 * dataW] + 8*src[sIdx + 2 + 3 * dataW] - 2*src[sIdx + 3 + 3 * dataW] - 2*src[sIdx + 4 + 3 * dataW] - 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] + src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 26 * gap] =  8*src[sIdx + 1 + dataW] - 8*src[sIdx + 2 + dataW] - 2*src[sIdx + 3 + dataW] + 2*src[sIdx + 4 + dataW] - 4*src[sIdx + 1 + 2 * dataW] + 4*src[sIdx + 2 + 2 * dataW] + src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] - 8*src[sIdx + 1 + 3 * dataW] + 8*src[sIdx + 2 + 3 * dataW] + 2*src[sIdx + 3 + 3 * dataW] - 2*src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 4*src[sIdx + 2 + 4 * dataW] - src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 27 * gap] =  -4*src[sIdx + 1 + dataW] - 2*src[sIdx + 2 + dataW] + 4*src[sIdx + 3 + dataW] + 2*src[sIdx + 4 + dataW] + 2*src[sIdx + 1 + 2 * dataW] + src[sIdx + 2 + 2 * dataW] - 2*src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] + 4*src[sIdx + 1 + 3 * dataW] + 2*src[sIdx + 2 + 3 * dataW] - 4*src[sIdx + 3 + 3 * dataW] - 2*src[sIdx + 4 + 3 * dataW] - 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] + 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 28 * gap] =  4*src[sIdx + 1 + dataW] - 2*src[sIdx + 2 + dataW] - 4*src[sIdx + 3 + dataW] + 2*src[sIdx + 4 + dataW] - 2*src[sIdx + 1 + 2 * dataW] + src[sIdx + 2 + 2 * dataW] + 2*src[sIdx + 3 + 2 * dataW] - src[sIdx + 4 + 2 * dataW] - 4*src[sIdx + 1 + 3 * dataW] + 2*src[sIdx + 2 + 3 * dataW] + 4*src[sIdx + 3 + 3 * dataW] - 2*src[sIdx + 4 + 3 * dataW] + 2*src[sIdx + 1 + 4 * dataW] - src[sIdx + 2 + 4 * dataW] - 2*src[sIdx + 3 + 4 * dataW] + src[sIdx + 4 + 4 * dataW];
    dst[bIdx + 29 * gap] =  8*src[sIdx + 1 + dataW] - 10*src[sIdx + 3 + dataW] + 2*src[sIdx + 5 + dataW] - 4*src[sIdx + 1 + 2 * dataW] + 5*src[sIdx + 3 + 2 * dataW] - src[sIdx + 5 + 2 * dataW] - 8*src[sIdx + 1 + 3 * dataW] + 10*src[sIdx + 3 + 3 * dataW] - 2*src[sIdx + 5 + 3 * dataW] + 4*src[sIdx + 1 + 4 * dataW] - 5*src[sIdx + 3 + 4 * dataW] + src[sIdx + 5 + 4 * dataW];

    dst[bIdx + 30 * gap] =  16*src[sIdx + dataW] - 20*src[sIdx + 2 + dataW] + 4*src[sIdx + 4 + dataW] - 20*src[sIdx + 3 * dataW] + 25*src[sIdx + 2 + 3 * dataW] - 5*src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 5 * dataW] - 5*src[sIdx + 2 + 5 * dataW] + src[sIdx + 4 + 5 * dataW];
    dst[bIdx + 31 * gap] =  -16*src[sIdx + 1 + dataW] - 16*src[sIdx + 2 + dataW] + 4*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] + 20*src[sIdx + 1 + 3 * dataW] + 20*src[sIdx + 2 + 3 * dataW] - 5*src[sIdx + 3 + 3 * dataW] - 5*src[sIdx + 4 + 3 * dataW] - 4*src[sIdx + 1 + 5 * dataW] - 4*src[sIdx + 2 + 5 * dataW] + src[sIdx + 3 + 5 * dataW] + src[sIdx + 4 + 5 * dataW];
    dst[bIdx + 32 * gap] =  16*src[sIdx + 1 + dataW] - 16*src[sIdx + 2 + dataW] - 4*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] - 20*src[sIdx + 1 + 3 * dataW] + 20*src[sIdx + 2 + 3 * dataW] + 5*src[sIdx + 3 + 3 * dataW] - 5*src[sIdx + 4 + 3 * dataW] + 4*src[sIdx + 1 + 5 * dataW] - 4*src[sIdx + 2 + 5 * dataW] - src[sIdx + 3 + 5 * dataW] + src[sIdx + 4 + 5 * dataW];
    dst[bIdx + 33 * gap] =  -8*src[sIdx + 1 + dataW] - 4*src[sIdx + 2 + dataW] + 8*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] + 10*src[sIdx + 1 + 3 * dataW] + 5*src[sIdx + 2 + 3 * dataW] - 10*src[sIdx + 3 + 3 * dataW] - 5*src[sIdx + 4 + 3 * dataW] - 2*src[sIdx + 1 + 5 * dataW] - src[sIdx + 2 + 5 * dataW] + 2*src[sIdx + 3 + 5 * dataW] + src[sIdx + 4 + 5 * dataW];
    dst[bIdx + 34 * gap] =  8*src[sIdx + 1 + dataW] - 4*src[sIdx + 2 + dataW] - 8*src[sIdx + 3 + dataW] + 4*src[sIdx + 4 + dataW] - 10*src[sIdx + 1 + 3 * dataW] + 5*src[sIdx + 2 + 3 * dataW] + 10*src[sIdx + 3 + 3 * dataW] - 5*src[sIdx + 4 + 3 * dataW] + 2*src[sIdx + 1 + 5 * dataW] - src[sIdx + 2 + 5 * dataW] - 2*src[sIdx + 3 + 5 * dataW] + src[sIdx + 4 + 5 * dataW];
    dst[bIdx + 35 * gap] =  16*src[sIdx + 1 + dataW] - 20*src[sIdx + 3 + dataW] + 4*src[sIdx + 5 + dataW] - 20*src[sIdx + 1 + 3 * dataW] + 25*src[sIdx + 3 + 3 * dataW] - 5*src[sIdx + 5 + 3 * dataW] + 4*src[sIdx + 1 + 5 * dataW] - 5*src[sIdx + 3 + 5 * dataW] + src[sIdx + 5 + 5 * dataW];
/*
    if (threadIdx.x + blockIdx.x + threadIdx.y + blockIdx.y == 0) {
      printf("original src:\n");
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          printf("%f ", src[sIdx + j + i * dataW]);
        }
        printf("\n");
      }
    

      printf("Src:");
      for (int i = 0; i < 36;i++) {
        if (i%6 == 0) printf("\n");
        printf("%f ", dst[bIdx + i * gap]);
      }
      printf("\n\n");
    }
    */
	}
}

template <typename Dtype> 
__global__ void wino6x6Src_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = tileH * tileW * inputs * batchs;
		int batchIdx = idx / (tileH * tileW * inputs);
		int inputIdx = (idx / (tileH * tileW)) % inputs ;
		int yIdx = (idx % (tileH * tileW)) / tileW ;
		int xIdx = idx % tileW;
		int bIdx =  idx % gap;
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 6 + xIdx * 6;

		//// -- project ---- ///
dst[bIdx +  0  * gap] =  src[sIdx + dataW * 0 + 0] - 21*src[sIdx + dataW * 0 + 2]/4 + 21*src[sIdx + dataW * 0 + 4]/4 - src[sIdx + dataW * 0 + 6] - 21*src[sIdx + dataW * 2 + 0]/4 + 441*src[sIdx + dataW * 2 + 2]/16 - 441*src[sIdx + dataW * 2 + 4]/16 + 21*src[sIdx + dataW * 2 + 6]/4 + 21*src[sIdx + dataW * 4 + 0]/4 - 441*src[sIdx + dataW * 4 + 2]/16 + 441*src[sIdx + dataW * 4 + 4]/16 - 21*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 6 + 0] + 21*src[sIdx + dataW * 6 + 2]/4 - 21*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  1  * gap] =  src[sIdx + dataW * 0 + 1] + src[sIdx + dataW * 0 + 2] - 17*src[sIdx + dataW * 0 + 3]/4 - 17*src[sIdx + dataW * 0 + 4]/4 + src[sIdx + dataW * 0 + 5] + src[sIdx + dataW * 0 + 6] - 21*src[sIdx + dataW * 2 + 1]/4 - 21*src[sIdx + dataW * 2 + 2]/4 + 357*src[sIdx + dataW * 2 + 3]/16 + 357*src[sIdx + dataW * 2 + 4]/16 - 21*src[sIdx + dataW * 2 + 5]/4 - 21*src[sIdx + dataW * 2 + 6]/4 + 21*src[sIdx + dataW * 4 + 1]/4 + 21*src[sIdx + dataW * 4 + 2]/4 - 357*src[sIdx + dataW * 4 + 3]/16 - 357*src[sIdx + dataW * 4 + 4]/16 + 21*src[sIdx + dataW * 4 + 5]/4 + 21*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 6 + 1] - src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 + 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] - src[sIdx + dataW * 6 + 6];
dst[bIdx +  2  * gap] =  -src[sIdx + dataW * 0 + 1] + src[sIdx + dataW * 0 + 2] + 17*src[sIdx + dataW * 0 + 3]/4 - 17*src[sIdx + dataW * 0 + 4]/4 - src[sIdx + dataW * 0 + 5] + src[sIdx + dataW * 0 + 6] + 21*src[sIdx + dataW * 2 + 1]/4 - 21*src[sIdx + dataW * 2 + 2]/4 - 357*src[sIdx + dataW * 2 + 3]/16 + 357*src[sIdx + dataW * 2 + 4]/16 + 21*src[sIdx + dataW * 2 + 5]/4 - 21*src[sIdx + dataW * 2 + 6]/4 - 21*src[sIdx + dataW * 4 + 1]/4 + 21*src[sIdx + dataW * 4 + 2]/4 + 357*src[sIdx + dataW * 4 + 3]/16 - 357*src[sIdx + dataW * 4 + 4]/16 - 21*src[sIdx + dataW * 4 + 5]/4 + 21*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 6 + 1] - src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 + 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] - src[sIdx + dataW * 6 + 6];
dst[bIdx +  3  * gap] =  src[sIdx + dataW * 0 + 1]/2 + src[sIdx + dataW * 0 + 2]/4 - 5*src[sIdx + dataW * 0 + 3]/2 - 5*src[sIdx + dataW * 0 + 4]/4 + 2*src[sIdx + dataW * 0 + 5] + src[sIdx + dataW * 0 + 6] - 21*src[sIdx + dataW * 2 + 1]/8 - 21*src[sIdx + dataW * 2 + 2]/16 + 105*src[sIdx + dataW * 2 + 3]/8 + 105*src[sIdx + dataW * 2 + 4]/16 - 21*src[sIdx + dataW * 2 + 5]/2 - 21*src[sIdx + dataW * 2 + 6]/4 + 21*src[sIdx + dataW * 4 + 1]/8 + 21*src[sIdx + dataW * 4 + 2]/16 - 105*src[sIdx + dataW * 4 + 3]/8 - 105*src[sIdx + dataW * 4 + 4]/16 + 21*src[sIdx + dataW * 4 + 5]/2 + 21*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 6 + 1]/2 - src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 + 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] - src[sIdx + dataW * 6 + 6];
dst[bIdx +  4  * gap] =  -src[sIdx + dataW * 0 + 1]/2 + src[sIdx + dataW * 0 + 2]/4 + 5*src[sIdx + dataW * 0 + 3]/2 - 5*src[sIdx + dataW * 0 + 4]/4 - 2*src[sIdx + dataW * 0 + 5] + src[sIdx + dataW * 0 + 6] + 21*src[sIdx + dataW * 2 + 1]/8 - 21*src[sIdx + dataW * 2 + 2]/16 - 105*src[sIdx + dataW * 2 + 3]/8 + 105*src[sIdx + dataW * 2 + 4]/16 + 21*src[sIdx + dataW * 2 + 5]/2 - 21*src[sIdx + dataW * 2 + 6]/4 - 21*src[sIdx + dataW * 4 + 1]/8 + 21*src[sIdx + dataW * 4 + 2]/16 + 105*src[sIdx + dataW * 4 + 3]/8 - 105*src[sIdx + dataW * 4 + 4]/16 - 21*src[sIdx + dataW * 4 + 5]/2 + 21*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 6 + 1]/2 - src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 + 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] - src[sIdx + dataW * 6 + 6];
dst[bIdx +  5  * gap] =  2*src[sIdx + dataW * 0 + 1] + 4*src[sIdx + dataW * 0 + 2] - 5*src[sIdx + dataW * 0 + 3]/2 - 5*src[sIdx + dataW * 0 + 4] + src[sIdx + dataW * 0 + 5]/2 + src[sIdx + dataW * 0 + 6] - 21*src[sIdx + dataW * 2 + 1]/2 - 21*src[sIdx + dataW * 2 + 2] + 105*src[sIdx + dataW * 2 + 3]/8 + 105*src[sIdx + dataW * 2 + 4]/4 - 21*src[sIdx + dataW * 2 + 5]/8 - 21*src[sIdx + dataW * 2 + 6]/4 + 21*src[sIdx + dataW * 4 + 1]/2 + 21*src[sIdx + dataW * 4 + 2] - 105*src[sIdx + dataW * 4 + 3]/8 - 105*src[sIdx + dataW * 4 + 4]/4 + 21*src[sIdx + dataW * 4 + 5]/8 + 21*src[sIdx + dataW * 4 + 6]/4 - 2*src[sIdx + dataW * 6 + 1] - 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 + 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  6  * gap] =  -2*src[sIdx + dataW * 0 + 1] + 4*src[sIdx + dataW * 0 + 2] + 5*src[sIdx + dataW * 0 + 3]/2 - 5*src[sIdx + dataW * 0 + 4] - src[sIdx + dataW * 0 + 5]/2 + src[sIdx + dataW * 0 + 6] + 21*src[sIdx + dataW * 2 + 1]/2 - 21*src[sIdx + dataW * 2 + 2] - 105*src[sIdx + dataW * 2 + 3]/8 + 105*src[sIdx + dataW * 2 + 4]/4 + 21*src[sIdx + dataW * 2 + 5]/8 - 21*src[sIdx + dataW * 2 + 6]/4 - 21*src[sIdx + dataW * 4 + 1]/2 + 21*src[sIdx + dataW * 4 + 2] + 105*src[sIdx + dataW * 4 + 3]/8 - 105*src[sIdx + dataW * 4 + 4]/4 - 21*src[sIdx + dataW * 4 + 5]/8 + 21*src[sIdx + dataW * 4 + 6]/4 + 2*src[sIdx + dataW * 6 + 1] - 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 + 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  7  * gap] =  -src[sIdx + dataW * 0 + 1] + 21*src[sIdx + dataW * 0 + 3]/4 - 21*src[sIdx + dataW * 0 + 5]/4 + src[sIdx + dataW * 0 + 7] + 21*src[sIdx + dataW * 2 + 1]/4 - 441*src[sIdx + dataW * 2 + 3]/16 + 441*src[sIdx + dataW * 2 + 5]/16 - 21*src[sIdx + dataW * 2 + 7]/4 - 21*src[sIdx + dataW * 4 + 1]/4 + 441*src[sIdx + dataW * 4 + 3]/16 - 441*src[sIdx + dataW * 4 + 5]/16 + 21*src[sIdx + dataW * 4 + 7]/4 + src[sIdx + dataW * 6 + 1] - 21*src[sIdx + dataW * 6 + 3]/4 + 21*src[sIdx + dataW * 6 + 5]/4 - src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  8  * gap] =  src[sIdx + dataW * 1 + 0] - 21*src[sIdx + dataW * 1 + 2]/4 + 21*src[sIdx + dataW * 1 + 4]/4 - src[sIdx + dataW * 1 + 6] + src[sIdx + dataW * 2 + 0] - 21*src[sIdx + dataW * 2 + 2]/4 + 21*src[sIdx + dataW * 2 + 4]/4 - src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 0]/4 + 357*src[sIdx + dataW * 3 + 2]/16 - 357*src[sIdx + dataW * 3 + 4]/16 + 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 0]/4 + 357*src[sIdx + dataW * 4 + 2]/16 - 357*src[sIdx + dataW * 4 + 4]/16 + 17*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 0] - 21*src[sIdx + dataW * 5 + 2]/4 + 21*src[sIdx + dataW * 5 + 4]/4 - src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 0] - 21*src[sIdx + dataW * 6 + 2]/4 + 21*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  9  * gap] =  src[sIdx + dataW * 1 + 1] + src[sIdx + dataW * 1 + 2] - 17*src[sIdx + dataW * 1 + 3]/4 - 17*src[sIdx + dataW * 1 + 4]/4 + src[sIdx + dataW * 1 + 5] + src[sIdx + dataW * 1 + 6] + src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] - 17*src[sIdx + dataW * 2 + 3]/4 - 17*src[sIdx + dataW * 2 + 4]/4 + src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 1]/4 - 17*src[sIdx + dataW * 3 + 2]/4 + 289*src[sIdx + dataW * 3 + 3]/16 + 289*src[sIdx + dataW * 3 + 4]/16 - 17*src[sIdx + dataW * 3 + 5]/4 - 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 1]/4 - 17*src[sIdx + dataW * 4 + 2]/4 + 289*src[sIdx + dataW * 4 + 3]/16 + 289*src[sIdx + dataW * 4 + 4]/16 - 17*src[sIdx + dataW * 4 + 5]/4 - 17*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 1] + src[sIdx + dataW * 5 + 2] - 17*src[sIdx + dataW * 5 + 3]/4 - 17*src[sIdx + dataW * 5 + 4]/4 + src[sIdx + dataW * 5 + 5] + src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  10  * gap] =  -src[sIdx + dataW * 1 + 1] + src[sIdx + dataW * 1 + 2] + 17*src[sIdx + dataW * 1 + 3]/4 - 17*src[sIdx + dataW * 1 + 4]/4 - src[sIdx + dataW * 1 + 5] + src[sIdx + dataW * 1 + 6] - src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] + 17*src[sIdx + dataW * 2 + 3]/4 - 17*src[sIdx + dataW * 2 + 4]/4 - src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 1]/4 - 17*src[sIdx + dataW * 3 + 2]/4 - 289*src[sIdx + dataW * 3 + 3]/16 + 289*src[sIdx + dataW * 3 + 4]/16 + 17*src[sIdx + dataW * 3 + 5]/4 - 17*src[sIdx + dataW * 3 + 6]/4 + 17*src[sIdx + dataW * 4 + 1]/4 - 17*src[sIdx + dataW * 4 + 2]/4 - 289*src[sIdx + dataW * 4 + 3]/16 + 289*src[sIdx + dataW * 4 + 4]/16 + 17*src[sIdx + dataW * 4 + 5]/4 - 17*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 1] + src[sIdx + dataW * 5 + 2] + 17*src[sIdx + dataW * 5 + 3]/4 - 17*src[sIdx + dataW * 5 + 4]/4 - src[sIdx + dataW * 5 + 5] + src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  11  * gap] =  src[sIdx + dataW * 1 + 1]/2 + src[sIdx + dataW * 1 + 2]/4 - 5*src[sIdx + dataW * 1 + 3]/2 - 5*src[sIdx + dataW * 1 + 4]/4 + 2*src[sIdx + dataW * 1 + 5] + src[sIdx + dataW * 1 + 6] + src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2]/4 - 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4]/4 + 2*src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 1]/8 - 17*src[sIdx + dataW * 3 + 2]/16 + 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/16 - 17*src[sIdx + dataW * 3 + 5]/2 - 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 1]/8 - 17*src[sIdx + dataW * 4 + 2]/16 + 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/16 - 17*src[sIdx + dataW * 4 + 5]/2 - 17*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 1]/2 + src[sIdx + dataW * 5 + 2]/4 - 5*src[sIdx + dataW * 5 + 3]/2 - 5*src[sIdx + dataW * 5 + 4]/4 + 2*src[sIdx + dataW * 5 + 5] + src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  12  * gap] =  -src[sIdx + dataW * 1 + 1]/2 + src[sIdx + dataW * 1 + 2]/4 + 5*src[sIdx + dataW * 1 + 3]/2 - 5*src[sIdx + dataW * 1 + 4]/4 - 2*src[sIdx + dataW * 1 + 5] + src[sIdx + dataW * 1 + 6] - src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2]/4 + 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4]/4 - 2*src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 1]/8 - 17*src[sIdx + dataW * 3 + 2]/16 - 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/16 + 17*src[sIdx + dataW * 3 + 5]/2 - 17*src[sIdx + dataW * 3 + 6]/4 + 17*src[sIdx + dataW * 4 + 1]/8 - 17*src[sIdx + dataW * 4 + 2]/16 - 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/16 + 17*src[sIdx + dataW * 4 + 5]/2 - 17*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 1]/2 + src[sIdx + dataW * 5 + 2]/4 + 5*src[sIdx + dataW * 5 + 3]/2 - 5*src[sIdx + dataW * 5 + 4]/4 - 2*src[sIdx + dataW * 5 + 5] + src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  13  * gap] =  2*src[sIdx + dataW * 1 + 1] + 4*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3]/2 - 5*src[sIdx + dataW * 1 + 4] + src[sIdx + dataW * 1 + 5]/2 + src[sIdx + dataW * 1 + 6] + 2*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] - 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4] + src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 1]/2 - 17*src[sIdx + dataW * 3 + 2] + 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/4 - 17*src[sIdx + dataW * 3 + 5]/8 - 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 1]/2 - 17*src[sIdx + dataW * 4 + 2] + 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/4 - 17*src[sIdx + dataW * 4 + 5]/8 - 17*src[sIdx + dataW * 4 + 6]/4 + 2*src[sIdx + dataW * 5 + 1] + 4*src[sIdx + dataW * 5 + 2] - 5*src[sIdx + dataW * 5 + 3]/2 - 5*src[sIdx + dataW * 5 + 4] + src[sIdx + dataW * 5 + 5]/2 + src[sIdx + dataW * 5 + 6] + 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  14  * gap] =  -2*src[sIdx + dataW * 1 + 1] + 4*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3]/2 - 5*src[sIdx + dataW * 1 + 4] - src[sIdx + dataW * 1 + 5]/2 + src[sIdx + dataW * 1 + 6] - 2*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] + 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4] - src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 1]/2 - 17*src[sIdx + dataW * 3 + 2] - 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/4 + 17*src[sIdx + dataW * 3 + 5]/8 - 17*src[sIdx + dataW * 3 + 6]/4 + 17*src[sIdx + dataW * 4 + 1]/2 - 17*src[sIdx + dataW * 4 + 2] - 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/4 + 17*src[sIdx + dataW * 4 + 5]/8 - 17*src[sIdx + dataW * 4 + 6]/4 - 2*src[sIdx + dataW * 5 + 1] + 4*src[sIdx + dataW * 5 + 2] + 5*src[sIdx + dataW * 5 + 3]/2 - 5*src[sIdx + dataW * 5 + 4] - src[sIdx + dataW * 5 + 5]/2 + src[sIdx + dataW * 5 + 6] - 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  15  * gap] =  -src[sIdx + dataW * 1 + 1] + 21*src[sIdx + dataW * 1 + 3]/4 - 21*src[sIdx + dataW * 1 + 5]/4 + src[sIdx + dataW * 1 + 7] - src[sIdx + dataW * 2 + 1] + 21*src[sIdx + dataW * 2 + 3]/4 - 21*src[sIdx + dataW * 2 + 5]/4 + src[sIdx + dataW * 2 + 7] + 17*src[sIdx + dataW * 3 + 1]/4 - 357*src[sIdx + dataW * 3 + 3]/16 + 357*src[sIdx + dataW * 3 + 5]/16 - 17*src[sIdx + dataW * 3 + 7]/4 + 17*src[sIdx + dataW * 4 + 1]/4 - 357*src[sIdx + dataW * 4 + 3]/16 + 357*src[sIdx + dataW * 4 + 5]/16 - 17*src[sIdx + dataW * 4 + 7]/4 - src[sIdx + dataW * 5 + 1] + 21*src[sIdx + dataW * 5 + 3]/4 - 21*src[sIdx + dataW * 5 + 5]/4 + src[sIdx + dataW * 5 + 7] - src[sIdx + dataW * 6 + 1] + 21*src[sIdx + dataW * 6 + 3]/4 - 21*src[sIdx + dataW * 6 + 5]/4 + src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  16  * gap] =  -src[sIdx + dataW * 1 + 0] + 21*src[sIdx + dataW * 1 + 2]/4 - 21*src[sIdx + dataW * 1 + 4]/4 + src[sIdx + dataW * 1 + 6] + src[sIdx + dataW * 2 + 0] - 21*src[sIdx + dataW * 2 + 2]/4 + 21*src[sIdx + dataW * 2 + 4]/4 - src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 0]/4 - 357*src[sIdx + dataW * 3 + 2]/16 + 357*src[sIdx + dataW * 3 + 4]/16 - 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 0]/4 + 357*src[sIdx + dataW * 4 + 2]/16 - 357*src[sIdx + dataW * 4 + 4]/16 + 17*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 0] + 21*src[sIdx + dataW * 5 + 2]/4 - 21*src[sIdx + dataW * 5 + 4]/4 + src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 0] - 21*src[sIdx + dataW * 6 + 2]/4 + 21*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  17  * gap] =  -src[sIdx + dataW * 1 + 1] - src[sIdx + dataW * 1 + 2] + 17*src[sIdx + dataW * 1 + 3]/4 + 17*src[sIdx + dataW * 1 + 4]/4 - src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] + src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] - 17*src[sIdx + dataW * 2 + 3]/4 - 17*src[sIdx + dataW * 2 + 4]/4 + src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 1]/4 + 17*src[sIdx + dataW * 3 + 2]/4 - 289*src[sIdx + dataW * 3 + 3]/16 - 289*src[sIdx + dataW * 3 + 4]/16 + 17*src[sIdx + dataW * 3 + 5]/4 + 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 1]/4 - 17*src[sIdx + dataW * 4 + 2]/4 + 289*src[sIdx + dataW * 4 + 3]/16 + 289*src[sIdx + dataW * 4 + 4]/16 - 17*src[sIdx + dataW * 4 + 5]/4 - 17*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 1] - src[sIdx + dataW * 5 + 2] + 17*src[sIdx + dataW * 5 + 3]/4 + 17*src[sIdx + dataW * 5 + 4]/4 - src[sIdx + dataW * 5 + 5] - src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  18  * gap] =  src[sIdx + dataW * 1 + 1] - src[sIdx + dataW * 1 + 2] - 17*src[sIdx + dataW * 1 + 3]/4 + 17*src[sIdx + dataW * 1 + 4]/4 + src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] - src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] + 17*src[sIdx + dataW * 2 + 3]/4 - 17*src[sIdx + dataW * 2 + 4]/4 - src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 1]/4 + 17*src[sIdx + dataW * 3 + 2]/4 + 289*src[sIdx + dataW * 3 + 3]/16 - 289*src[sIdx + dataW * 3 + 4]/16 - 17*src[sIdx + dataW * 3 + 5]/4 + 17*src[sIdx + dataW * 3 + 6]/4 + 17*src[sIdx + dataW * 4 + 1]/4 - 17*src[sIdx + dataW * 4 + 2]/4 - 289*src[sIdx + dataW * 4 + 3]/16 + 289*src[sIdx + dataW * 4 + 4]/16 + 17*src[sIdx + dataW * 4 + 5]/4 - 17*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 1] - src[sIdx + dataW * 5 + 2] - 17*src[sIdx + dataW * 5 + 3]/4 + 17*src[sIdx + dataW * 5 + 4]/4 + src[sIdx + dataW * 5 + 5] - src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  19  * gap] =  -src[sIdx + dataW * 1 + 1]/2 - src[sIdx + dataW * 1 + 2]/4 + 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4]/4 - 2*src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] + src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2]/4 - 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4]/4 + 2*src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 1]/8 + 17*src[sIdx + dataW * 3 + 2]/16 - 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/16 + 17*src[sIdx + dataW * 3 + 5]/2 + 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 1]/8 - 17*src[sIdx + dataW * 4 + 2]/16 + 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/16 - 17*src[sIdx + dataW * 4 + 5]/2 - 17*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 1]/2 - src[sIdx + dataW * 5 + 2]/4 + 5*src[sIdx + dataW * 5 + 3]/2 + 5*src[sIdx + dataW * 5 + 4]/4 - 2*src[sIdx + dataW * 5 + 5] - src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  20  * gap] =  src[sIdx + dataW * 1 + 1]/2 - src[sIdx + dataW * 1 + 2]/4 - 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4]/4 + 2*src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] - src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2]/4 + 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4]/4 - 2*src[sIdx + dataW * 2 + 5] + src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 1]/8 + 17*src[sIdx + dataW * 3 + 2]/16 + 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/16 - 17*src[sIdx + dataW * 3 + 5]/2 + 17*src[sIdx + dataW * 3 + 6]/4 + 17*src[sIdx + dataW * 4 + 1]/8 - 17*src[sIdx + dataW * 4 + 2]/16 - 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/16 + 17*src[sIdx + dataW * 4 + 5]/2 - 17*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 1]/2 - src[sIdx + dataW * 5 + 2]/4 - 5*src[sIdx + dataW * 5 + 3]/2 + 5*src[sIdx + dataW * 5 + 4]/4 + 2*src[sIdx + dataW * 5 + 5] - src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  21  * gap] =  -2*src[sIdx + dataW * 1 + 1] - 4*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4] - src[sIdx + dataW * 1 + 5]/2 - src[sIdx + dataW * 1 + 6] + 2*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] - 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4] + src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6] + 17*src[sIdx + dataW * 3 + 1]/2 + 17*src[sIdx + dataW * 3 + 2] - 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/4 + 17*src[sIdx + dataW * 3 + 5]/8 + 17*src[sIdx + dataW * 3 + 6]/4 - 17*src[sIdx + dataW * 4 + 1]/2 - 17*src[sIdx + dataW * 4 + 2] + 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/4 - 17*src[sIdx + dataW * 4 + 5]/8 - 17*src[sIdx + dataW * 4 + 6]/4 - 2*src[sIdx + dataW * 5 + 1] - 4*src[sIdx + dataW * 5 + 2] + 5*src[sIdx + dataW * 5 + 3]/2 + 5*src[sIdx + dataW * 5 + 4] - src[sIdx + dataW * 5 + 5]/2 - src[sIdx + dataW * 5 + 6] + 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  22  * gap] =  2*src[sIdx + dataW * 1 + 1] - 4*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4] + src[sIdx + dataW * 1 + 5]/2 - src[sIdx + dataW * 1 + 6] - 2*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] + 5*src[sIdx + dataW * 2 + 3]/2 - 5*src[sIdx + dataW * 2 + 4] - src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6] - 17*src[sIdx + dataW * 3 + 1]/2 + 17*src[sIdx + dataW * 3 + 2] + 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/4 - 17*src[sIdx + dataW * 3 + 5]/8 + 17*src[sIdx + dataW * 3 + 6]/4 + 17*src[sIdx + dataW * 4 + 1]/2 - 17*src[sIdx + dataW * 4 + 2] - 85*src[sIdx + dataW * 4 + 3]/8 + 85*src[sIdx + dataW * 4 + 4]/4 + 17*src[sIdx + dataW * 4 + 5]/8 - 17*src[sIdx + dataW * 4 + 6]/4 + 2*src[sIdx + dataW * 5 + 1] - 4*src[sIdx + dataW * 5 + 2] - 5*src[sIdx + dataW * 5 + 3]/2 + 5*src[sIdx + dataW * 5 + 4] + src[sIdx + dataW * 5 + 5]/2 - src[sIdx + dataW * 5 + 6] - 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  23  * gap] =  src[sIdx + dataW * 1 + 1] - 21*src[sIdx + dataW * 1 + 3]/4 + 21*src[sIdx + dataW * 1 + 5]/4 - src[sIdx + dataW * 1 + 7] - src[sIdx + dataW * 2 + 1] + 21*src[sIdx + dataW * 2 + 3]/4 - 21*src[sIdx + dataW * 2 + 5]/4 + src[sIdx + dataW * 2 + 7] - 17*src[sIdx + dataW * 3 + 1]/4 + 357*src[sIdx + dataW * 3 + 3]/16 - 357*src[sIdx + dataW * 3 + 5]/16 + 17*src[sIdx + dataW * 3 + 7]/4 + 17*src[sIdx + dataW * 4 + 1]/4 - 357*src[sIdx + dataW * 4 + 3]/16 + 357*src[sIdx + dataW * 4 + 5]/16 - 17*src[sIdx + dataW * 4 + 7]/4 + src[sIdx + dataW * 5 + 1] - 21*src[sIdx + dataW * 5 + 3]/4 + 21*src[sIdx + dataW * 5 + 5]/4 - src[sIdx + dataW * 5 + 7] - src[sIdx + dataW * 6 + 1] + 21*src[sIdx + dataW * 6 + 3]/4 - 21*src[sIdx + dataW * 6 + 5]/4 + src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  24  * gap] =  src[sIdx + dataW * 1 + 0]/2 - 21*src[sIdx + dataW * 1 + 2]/8 + 21*src[sIdx + dataW * 1 + 4]/8 - src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 0]/4 - 21*src[sIdx + dataW * 2 + 2]/16 + 21*src[sIdx + dataW * 2 + 4]/16 - src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 0]/2 + 105*src[sIdx + dataW * 3 + 2]/8 - 105*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 0]/4 + 105*src[sIdx + dataW * 4 + 2]/16 - 105*src[sIdx + dataW * 4 + 4]/16 + 5*src[sIdx + dataW * 4 + 6]/4 + 2*src[sIdx + dataW * 5 + 0] - 21*src[sIdx + dataW * 5 + 2]/2 + 21*src[sIdx + dataW * 5 + 4]/2 - 2*src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 0] - 21*src[sIdx + dataW * 6 + 2]/4 + 21*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  25  * gap] =  src[sIdx + dataW * 1 + 1]/2 + src[sIdx + dataW * 1 + 2]/2 - 17*src[sIdx + dataW * 1 + 3]/8 - 17*src[sIdx + dataW * 1 + 4]/8 + src[sIdx + dataW * 1 + 5]/2 + src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 1]/4 + src[sIdx + dataW * 2 + 2]/4 - 17*src[sIdx + dataW * 2 + 3]/16 - 17*src[sIdx + dataW * 2 + 4]/16 + src[sIdx + dataW * 2 + 5]/4 + src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 1]/2 - 5*src[sIdx + dataW * 3 + 2]/2 + 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5]/2 - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/4 - 5*src[sIdx + dataW * 4 + 2]/4 + 85*src[sIdx + dataW * 4 + 3]/16 + 85*src[sIdx + dataW * 4 + 4]/16 - 5*src[sIdx + dataW * 4 + 5]/4 - 5*src[sIdx + dataW * 4 + 6]/4 + 2*src[sIdx + dataW * 5 + 1] + 2*src[sIdx + dataW * 5 + 2] - 17*src[sIdx + dataW * 5 + 3]/2 - 17*src[sIdx + dataW * 5 + 4]/2 + 2*src[sIdx + dataW * 5 + 5] + 2*src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  26  * gap] =  -src[sIdx + dataW * 1 + 1]/2 + src[sIdx + dataW * 1 + 2]/2 + 17*src[sIdx + dataW * 1 + 3]/8 - 17*src[sIdx + dataW * 1 + 4]/8 - src[sIdx + dataW * 1 + 5]/2 + src[sIdx + dataW * 1 + 6]/2 - src[sIdx + dataW * 2 + 1]/4 + src[sIdx + dataW * 2 + 2]/4 + 17*src[sIdx + dataW * 2 + 3]/16 - 17*src[sIdx + dataW * 2 + 4]/16 - src[sIdx + dataW * 2 + 5]/4 + src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 1]/2 - 5*src[sIdx + dataW * 3 + 2]/2 - 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5]/2 - 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/4 - 5*src[sIdx + dataW * 4 + 2]/4 - 85*src[sIdx + dataW * 4 + 3]/16 + 85*src[sIdx + dataW * 4 + 4]/16 + 5*src[sIdx + dataW * 4 + 5]/4 - 5*src[sIdx + dataW * 4 + 6]/4 - 2*src[sIdx + dataW * 5 + 1] + 2*src[sIdx + dataW * 5 + 2] + 17*src[sIdx + dataW * 5 + 3]/2 - 17*src[sIdx + dataW * 5 + 4]/2 - 2*src[sIdx + dataW * 5 + 5] + 2*src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  27  * gap] =  src[sIdx + dataW * 1 + 1]/4 + src[sIdx + dataW * 1 + 2]/8 - 5*src[sIdx + dataW * 1 + 3]/4 - 5*src[sIdx + dataW * 1 + 4]/8 + src[sIdx + dataW * 1 + 5] + src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 1]/8 + src[sIdx + dataW * 2 + 2]/16 - 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/16 + src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 1]/4 - 5*src[sIdx + dataW * 3 + 2]/8 + 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5] - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/8 - 5*src[sIdx + dataW * 4 + 2]/16 + 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/16 - 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 1] + src[sIdx + dataW * 5 + 2]/2 - 5*src[sIdx + dataW * 5 + 3] - 5*src[sIdx + dataW * 5 + 4]/2 + 4*src[sIdx + dataW * 5 + 5] + 2*src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  28  * gap] =  -src[sIdx + dataW * 1 + 1]/4 + src[sIdx + dataW * 1 + 2]/8 + 5*src[sIdx + dataW * 1 + 3]/4 - 5*src[sIdx + dataW * 1 + 4]/8 - src[sIdx + dataW * 1 + 5] + src[sIdx + dataW * 1 + 6]/2 - src[sIdx + dataW * 2 + 1]/8 + src[sIdx + dataW * 2 + 2]/16 + 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/16 - src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 1]/4 - 5*src[sIdx + dataW * 3 + 2]/8 - 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5] - 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/8 - 5*src[sIdx + dataW * 4 + 2]/16 - 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/16 + 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 1] + src[sIdx + dataW * 5 + 2]/2 + 5*src[sIdx + dataW * 5 + 3] - 5*src[sIdx + dataW * 5 + 4]/2 - 4*src[sIdx + dataW * 5 + 5] + 2*src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  29  * gap] =  src[sIdx + dataW * 1 + 1] + 2*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3]/4 - 5*src[sIdx + dataW * 1 + 4]/2 + src[sIdx + dataW * 1 + 5]/4 + src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2] - 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/4 + src[sIdx + dataW * 2 + 5]/8 + src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 1] - 10*src[sIdx + dataW * 3 + 2] + 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/2 - 5*src[sIdx + dataW * 3 + 5]/4 - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2] + 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/4 - 5*src[sIdx + dataW * 4 + 5]/8 - 5*src[sIdx + dataW * 4 + 6]/4 + 4*src[sIdx + dataW * 5 + 1] + 8*src[sIdx + dataW * 5 + 2] - 5*src[sIdx + dataW * 5 + 3] - 10*src[sIdx + dataW * 5 + 4] + src[sIdx + dataW * 5 + 5] + 2*src[sIdx + dataW * 5 + 6] + 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  30  * gap] =  -src[sIdx + dataW * 1 + 1] + 2*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3]/4 - 5*src[sIdx + dataW * 1 + 4]/2 - src[sIdx + dataW * 1 + 5]/4 + src[sIdx + dataW * 1 + 6]/2 - src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2] + 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/4 - src[sIdx + dataW * 2 + 5]/8 + src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 1] - 10*src[sIdx + dataW * 3 + 2] - 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/2 + 5*src[sIdx + dataW * 3 + 5]/4 - 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2] - 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/4 + 5*src[sIdx + dataW * 4 + 5]/8 - 5*src[sIdx + dataW * 4 + 6]/4 - 4*src[sIdx + dataW * 5 + 1] + 8*src[sIdx + dataW * 5 + 2] + 5*src[sIdx + dataW * 5 + 3] - 10*src[sIdx + dataW * 5 + 4] - src[sIdx + dataW * 5 + 5] + 2*src[sIdx + dataW * 5 + 6] - 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  31  * gap] =  -src[sIdx + dataW * 1 + 1]/2 + 21*src[sIdx + dataW * 1 + 3]/8 - 21*src[sIdx + dataW * 1 + 5]/8 + src[sIdx + dataW * 1 + 7]/2 - src[sIdx + dataW * 2 + 1]/4 + 21*src[sIdx + dataW * 2 + 3]/16 - 21*src[sIdx + dataW * 2 + 5]/16 + src[sIdx + dataW * 2 + 7]/4 + 5*src[sIdx + dataW * 3 + 1]/2 - 105*src[sIdx + dataW * 3 + 3]/8 + 105*src[sIdx + dataW * 3 + 5]/8 - 5*src[sIdx + dataW * 3 + 7]/2 + 5*src[sIdx + dataW * 4 + 1]/4 - 105*src[sIdx + dataW * 4 + 3]/16 + 105*src[sIdx + dataW * 4 + 5]/16 - 5*src[sIdx + dataW * 4 + 7]/4 - 2*src[sIdx + dataW * 5 + 1] + 21*src[sIdx + dataW * 5 + 3]/2 - 21*src[sIdx + dataW * 5 + 5]/2 + 2*src[sIdx + dataW * 5 + 7] - src[sIdx + dataW * 6 + 1] + 21*src[sIdx + dataW * 6 + 3]/4 - 21*src[sIdx + dataW * 6 + 5]/4 + src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  32  * gap] =  -src[sIdx + dataW * 1 + 0]/2 + 21*src[sIdx + dataW * 1 + 2]/8 - 21*src[sIdx + dataW * 1 + 4]/8 + src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 0]/4 - 21*src[sIdx + dataW * 2 + 2]/16 + 21*src[sIdx + dataW * 2 + 4]/16 - src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 0]/2 - 105*src[sIdx + dataW * 3 + 2]/8 + 105*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 0]/4 + 105*src[sIdx + dataW * 4 + 2]/16 - 105*src[sIdx + dataW * 4 + 4]/16 + 5*src[sIdx + dataW * 4 + 6]/4 - 2*src[sIdx + dataW * 5 + 0] + 21*src[sIdx + dataW * 5 + 2]/2 - 21*src[sIdx + dataW * 5 + 4]/2 + 2*src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 0] - 21*src[sIdx + dataW * 6 + 2]/4 + 21*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  33  * gap] =  -src[sIdx + dataW * 1 + 1]/2 - src[sIdx + dataW * 1 + 2]/2 + 17*src[sIdx + dataW * 1 + 3]/8 + 17*src[sIdx + dataW * 1 + 4]/8 - src[sIdx + dataW * 1 + 5]/2 - src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 1]/4 + src[sIdx + dataW * 2 + 2]/4 - 17*src[sIdx + dataW * 2 + 3]/16 - 17*src[sIdx + dataW * 2 + 4]/16 + src[sIdx + dataW * 2 + 5]/4 + src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 1]/2 + 5*src[sIdx + dataW * 3 + 2]/2 - 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5]/2 + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/4 - 5*src[sIdx + dataW * 4 + 2]/4 + 85*src[sIdx + dataW * 4 + 3]/16 + 85*src[sIdx + dataW * 4 + 4]/16 - 5*src[sIdx + dataW * 4 + 5]/4 - 5*src[sIdx + dataW * 4 + 6]/4 - 2*src[sIdx + dataW * 5 + 1] - 2*src[sIdx + dataW * 5 + 2] + 17*src[sIdx + dataW * 5 + 3]/2 + 17*src[sIdx + dataW * 5 + 4]/2 - 2*src[sIdx + dataW * 5 + 5] - 2*src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  34  * gap] =  src[sIdx + dataW * 1 + 1]/2 - src[sIdx + dataW * 1 + 2]/2 - 17*src[sIdx + dataW * 1 + 3]/8 + 17*src[sIdx + dataW * 1 + 4]/8 + src[sIdx + dataW * 1 + 5]/2 - src[sIdx + dataW * 1 + 6]/2 - src[sIdx + dataW * 2 + 1]/4 + src[sIdx + dataW * 2 + 2]/4 + 17*src[sIdx + dataW * 2 + 3]/16 - 17*src[sIdx + dataW * 2 + 4]/16 - src[sIdx + dataW * 2 + 5]/4 + src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 1]/2 + 5*src[sIdx + dataW * 3 + 2]/2 + 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5]/2 + 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/4 - 5*src[sIdx + dataW * 4 + 2]/4 - 85*src[sIdx + dataW * 4 + 3]/16 + 85*src[sIdx + dataW * 4 + 4]/16 + 5*src[sIdx + dataW * 4 + 5]/4 - 5*src[sIdx + dataW * 4 + 6]/4 + 2*src[sIdx + dataW * 5 + 1] - 2*src[sIdx + dataW * 5 + 2] - 17*src[sIdx + dataW * 5 + 3]/2 + 17*src[sIdx + dataW * 5 + 4]/2 + 2*src[sIdx + dataW * 5 + 5] - 2*src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  35  * gap] =  -src[sIdx + dataW * 1 + 1]/4 - src[sIdx + dataW * 1 + 2]/8 + 5*src[sIdx + dataW * 1 + 3]/4 + 5*src[sIdx + dataW * 1 + 4]/8 - src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 1]/8 + src[sIdx + dataW * 2 + 2]/16 - 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/16 + src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 1]/4 + 5*src[sIdx + dataW * 3 + 2]/8 - 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5] + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/8 - 5*src[sIdx + dataW * 4 + 2]/16 + 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/16 - 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6]/4 - src[sIdx + dataW * 5 + 1] - src[sIdx + dataW * 5 + 2]/2 + 5*src[sIdx + dataW * 5 + 3] + 5*src[sIdx + dataW * 5 + 4]/2 - 4*src[sIdx + dataW * 5 + 5] - 2*src[sIdx + dataW * 5 + 6] + src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  36  * gap] =  src[sIdx + dataW * 1 + 1]/4 - src[sIdx + dataW * 1 + 2]/8 - 5*src[sIdx + dataW * 1 + 3]/4 + 5*src[sIdx + dataW * 1 + 4]/8 + src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6]/2 - src[sIdx + dataW * 2 + 1]/8 + src[sIdx + dataW * 2 + 2]/16 + 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/16 - src[sIdx + dataW * 2 + 5]/2 + src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 1]/4 + 5*src[sIdx + dataW * 3 + 2]/8 + 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5] + 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/8 - 5*src[sIdx + dataW * 4 + 2]/16 - 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/16 + 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6]/4 + src[sIdx + dataW * 5 + 1] - src[sIdx + dataW * 5 + 2]/2 - 5*src[sIdx + dataW * 5 + 3] + 5*src[sIdx + dataW * 5 + 4]/2 + 4*src[sIdx + dataW * 5 + 5] - 2*src[sIdx + dataW * 5 + 6] - src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  37  * gap] =  -src[sIdx + dataW * 1 + 1] - 2*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3]/4 + 5*src[sIdx + dataW * 1 + 4]/2 - src[sIdx + dataW * 1 + 5]/4 - src[sIdx + dataW * 1 + 6]/2 + src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2] - 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/4 + src[sIdx + dataW * 2 + 5]/8 + src[sIdx + dataW * 2 + 6]/4 + 5*src[sIdx + dataW * 3 + 1] + 10*src[sIdx + dataW * 3 + 2] - 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/2 + 5*src[sIdx + dataW * 3 + 5]/4 + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2] + 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/4 - 5*src[sIdx + dataW * 4 + 5]/8 - 5*src[sIdx + dataW * 4 + 6]/4 - 4*src[sIdx + dataW * 5 + 1] - 8*src[sIdx + dataW * 5 + 2] + 5*src[sIdx + dataW * 5 + 3] + 10*src[sIdx + dataW * 5 + 4] - src[sIdx + dataW * 5 + 5] - 2*src[sIdx + dataW * 5 + 6] + 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  38  * gap] =  src[sIdx + dataW * 1 + 1] - 2*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3]/4 + 5*src[sIdx + dataW * 1 + 4]/2 + src[sIdx + dataW * 1 + 5]/4 - src[sIdx + dataW * 1 + 6]/2 - src[sIdx + dataW * 2 + 1]/2 + src[sIdx + dataW * 2 + 2] + 5*src[sIdx + dataW * 2 + 3]/8 - 5*src[sIdx + dataW * 2 + 4]/4 - src[sIdx + dataW * 2 + 5]/8 + src[sIdx + dataW * 2 + 6]/4 - 5*src[sIdx + dataW * 3 + 1] + 10*src[sIdx + dataW * 3 + 2] + 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/2 - 5*src[sIdx + dataW * 3 + 5]/4 + 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2] - 25*src[sIdx + dataW * 4 + 3]/8 + 25*src[sIdx + dataW * 4 + 4]/4 + 5*src[sIdx + dataW * 4 + 5]/8 - 5*src[sIdx + dataW * 4 + 6]/4 + 4*src[sIdx + dataW * 5 + 1] - 8*src[sIdx + dataW * 5 + 2] - 5*src[sIdx + dataW * 5 + 3] + 10*src[sIdx + dataW * 5 + 4] + src[sIdx + dataW * 5 + 5] - 2*src[sIdx + dataW * 5 + 6] - 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  39  * gap] =  src[sIdx + dataW * 1 + 1]/2 - 21*src[sIdx + dataW * 1 + 3]/8 + 21*src[sIdx + dataW * 1 + 5]/8 - src[sIdx + dataW * 1 + 7]/2 - src[sIdx + dataW * 2 + 1]/4 + 21*src[sIdx + dataW * 2 + 3]/16 - 21*src[sIdx + dataW * 2 + 5]/16 + src[sIdx + dataW * 2 + 7]/4 - 5*src[sIdx + dataW * 3 + 1]/2 + 105*src[sIdx + dataW * 3 + 3]/8 - 105*src[sIdx + dataW * 3 + 5]/8 + 5*src[sIdx + dataW * 3 + 7]/2 + 5*src[sIdx + dataW * 4 + 1]/4 - 105*src[sIdx + dataW * 4 + 3]/16 + 105*src[sIdx + dataW * 4 + 5]/16 - 5*src[sIdx + dataW * 4 + 7]/4 + 2*src[sIdx + dataW * 5 + 1] - 21*src[sIdx + dataW * 5 + 3]/2 + 21*src[sIdx + dataW * 5 + 5]/2 - 2*src[sIdx + dataW * 5 + 7] - src[sIdx + dataW * 6 + 1] + 21*src[sIdx + dataW * 6 + 3]/4 - 21*src[sIdx + dataW * 6 + 5]/4 + src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  40  * gap] =  2*src[sIdx + dataW * 1 + 0] - 21*src[sIdx + dataW * 1 + 2]/2 + 21*src[sIdx + dataW * 1 + 4]/2 - 2*src[sIdx + dataW * 1 + 6] + 4*src[sIdx + dataW * 2 + 0] - 21*src[sIdx + dataW * 2 + 2] + 21*src[sIdx + dataW * 2 + 4] - 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 0]/2 + 105*src[sIdx + dataW * 3 + 2]/8 - 105*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 0] + 105*src[sIdx + dataW * 4 + 2]/4 - 105*src[sIdx + dataW * 4 + 4]/4 + 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 0]/2 - 21*src[sIdx + dataW * 5 + 2]/8 + 21*src[sIdx + dataW * 5 + 4]/8 - src[sIdx + dataW * 5 + 6]/2 + src[sIdx + dataW * 6 + 0] - 21*src[sIdx + dataW * 6 + 2]/4 + 21*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  41  * gap] =  2*src[sIdx + dataW * 1 + 1] + 2*src[sIdx + dataW * 1 + 2] - 17*src[sIdx + dataW * 1 + 3]/2 - 17*src[sIdx + dataW * 1 + 4]/2 + 2*src[sIdx + dataW * 1 + 5] + 2*src[sIdx + dataW * 1 + 6] + 4*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] - 17*src[sIdx + dataW * 2 + 3] - 17*src[sIdx + dataW * 2 + 4] + 4*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 1]/2 - 5*src[sIdx + dataW * 3 + 2]/2 + 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5]/2 - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1] - 5*src[sIdx + dataW * 4 + 2] + 85*src[sIdx + dataW * 4 + 3]/4 + 85*src[sIdx + dataW * 4 + 4]/4 - 5*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 1]/2 + src[sIdx + dataW * 5 + 2]/2 - 17*src[sIdx + dataW * 5 + 3]/8 - 17*src[sIdx + dataW * 5 + 4]/8 + src[sIdx + dataW * 5 + 5]/2 + src[sIdx + dataW * 5 + 6]/2 + src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  42  * gap] =  -2*src[sIdx + dataW * 1 + 1] + 2*src[sIdx + dataW * 1 + 2] + 17*src[sIdx + dataW * 1 + 3]/2 - 17*src[sIdx + dataW * 1 + 4]/2 - 2*src[sIdx + dataW * 1 + 5] + 2*src[sIdx + dataW * 1 + 6] - 4*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] + 17*src[sIdx + dataW * 2 + 3] - 17*src[sIdx + dataW * 2 + 4] - 4*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 1]/2 - 5*src[sIdx + dataW * 3 + 2]/2 - 85*src[sIdx + dataW * 3 + 3]/8 + 85*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5]/2 - 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1] - 5*src[sIdx + dataW * 4 + 2] - 85*src[sIdx + dataW * 4 + 3]/4 + 85*src[sIdx + dataW * 4 + 4]/4 + 5*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 1]/2 + src[sIdx + dataW * 5 + 2]/2 + 17*src[sIdx + dataW * 5 + 3]/8 - 17*src[sIdx + dataW * 5 + 4]/8 - src[sIdx + dataW * 5 + 5]/2 + src[sIdx + dataW * 5 + 6]/2 - src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  43  * gap] =  src[sIdx + dataW * 1 + 1] + src[sIdx + dataW * 1 + 2]/2 - 5*src[sIdx + dataW * 1 + 3] - 5*src[sIdx + dataW * 1 + 4]/2 + 4*src[sIdx + dataW * 1 + 5] + 2*src[sIdx + dataW * 1 + 6] + 2*src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] - 10*src[sIdx + dataW * 2 + 3] - 5*src[sIdx + dataW * 2 + 4] + 8*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 1]/4 - 5*src[sIdx + dataW * 3 + 2]/8 + 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5] - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2]/4 + 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4]/4 - 10*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 1]/4 + src[sIdx + dataW * 5 + 2]/8 - 5*src[sIdx + dataW * 5 + 3]/4 - 5*src[sIdx + dataW * 5 + 4]/8 + src[sIdx + dataW * 5 + 5] + src[sIdx + dataW * 5 + 6]/2 + src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  44  * gap] =  -src[sIdx + dataW * 1 + 1] + src[sIdx + dataW * 1 + 2]/2 + 5*src[sIdx + dataW * 1 + 3] - 5*src[sIdx + dataW * 1 + 4]/2 - 4*src[sIdx + dataW * 1 + 5] + 2*src[sIdx + dataW * 1 + 6] - 2*src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] + 10*src[sIdx + dataW * 2 + 3] - 5*src[sIdx + dataW * 2 + 4] - 8*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 1]/4 - 5*src[sIdx + dataW * 3 + 2]/8 - 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5] - 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2]/4 - 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4]/4 + 10*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 1]/4 + src[sIdx + dataW * 5 + 2]/8 + 5*src[sIdx + dataW * 5 + 3]/4 - 5*src[sIdx + dataW * 5 + 4]/8 - src[sIdx + dataW * 5 + 5] + src[sIdx + dataW * 5 + 6]/2 - src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  45  * gap] =  4*src[sIdx + dataW * 1 + 1] + 8*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3] - 10*src[sIdx + dataW * 1 + 4] + src[sIdx + dataW * 1 + 5] + 2*src[sIdx + dataW * 1 + 6] + 8*src[sIdx + dataW * 2 + 1] + 16*src[sIdx + dataW * 2 + 2] - 10*src[sIdx + dataW * 2 + 3] - 20*src[sIdx + dataW * 2 + 4] + 2*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 1] - 10*src[sIdx + dataW * 3 + 2] + 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/2 - 5*src[sIdx + dataW * 3 + 5]/4 - 5*src[sIdx + dataW * 3 + 6]/2 - 10*src[sIdx + dataW * 4 + 1] - 20*src[sIdx + dataW * 4 + 2] + 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4] - 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 1] + 2*src[sIdx + dataW * 5 + 2] - 5*src[sIdx + dataW * 5 + 3]/4 - 5*src[sIdx + dataW * 5 + 4]/2 + src[sIdx + dataW * 5 + 5]/4 + src[sIdx + dataW * 5 + 6]/2 + 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  46  * gap] =  -4*src[sIdx + dataW * 1 + 1] + 8*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3] - 10*src[sIdx + dataW * 1 + 4] - src[sIdx + dataW * 1 + 5] + 2*src[sIdx + dataW * 1 + 6] - 8*src[sIdx + dataW * 2 + 1] + 16*src[sIdx + dataW * 2 + 2] + 10*src[sIdx + dataW * 2 + 3] - 20*src[sIdx + dataW * 2 + 4] - 2*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 1] - 10*src[sIdx + dataW * 3 + 2] - 25*src[sIdx + dataW * 3 + 3]/4 + 25*src[sIdx + dataW * 3 + 4]/2 + 5*src[sIdx + dataW * 3 + 5]/4 - 5*src[sIdx + dataW * 3 + 6]/2 + 10*src[sIdx + dataW * 4 + 1] - 20*src[sIdx + dataW * 4 + 2] - 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4] + 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 1] + 2*src[sIdx + dataW * 5 + 2] + 5*src[sIdx + dataW * 5 + 3]/4 - 5*src[sIdx + dataW * 5 + 4]/2 - src[sIdx + dataW * 5 + 5]/4 + src[sIdx + dataW * 5 + 6]/2 - 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  47  * gap] =  -2*src[sIdx + dataW * 1 + 1] + 21*src[sIdx + dataW * 1 + 3]/2 - 21*src[sIdx + dataW * 1 + 5]/2 + 2*src[sIdx + dataW * 1 + 7] - 4*src[sIdx + dataW * 2 + 1] + 21*src[sIdx + dataW * 2 + 3] - 21*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 7] + 5*src[sIdx + dataW * 3 + 1]/2 - 105*src[sIdx + dataW * 3 + 3]/8 + 105*src[sIdx + dataW * 3 + 5]/8 - 5*src[sIdx + dataW * 3 + 7]/2 + 5*src[sIdx + dataW * 4 + 1] - 105*src[sIdx + dataW * 4 + 3]/4 + 105*src[sIdx + dataW * 4 + 5]/4 - 5*src[sIdx + dataW * 4 + 7] - src[sIdx + dataW * 5 + 1]/2 + 21*src[sIdx + dataW * 5 + 3]/8 - 21*src[sIdx + dataW * 5 + 5]/8 + src[sIdx + dataW * 5 + 7]/2 - src[sIdx + dataW * 6 + 1] + 21*src[sIdx + dataW * 6 + 3]/4 - 21*src[sIdx + dataW * 6 + 5]/4 + src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  48  * gap] =  -2*src[sIdx + dataW * 1 + 0] + 21*src[sIdx + dataW * 1 + 2]/2 - 21*src[sIdx + dataW * 1 + 4]/2 + 2*src[sIdx + dataW * 1 + 6] + 4*src[sIdx + dataW * 2 + 0] - 21*src[sIdx + dataW * 2 + 2] + 21*src[sIdx + dataW * 2 + 4] - 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 0]/2 - 105*src[sIdx + dataW * 3 + 2]/8 + 105*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 0] + 105*src[sIdx + dataW * 4 + 2]/4 - 105*src[sIdx + dataW * 4 + 4]/4 + 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 0]/2 + 21*src[sIdx + dataW * 5 + 2]/8 - 21*src[sIdx + dataW * 5 + 4]/8 + src[sIdx + dataW * 5 + 6]/2 + src[sIdx + dataW * 6 + 0] - 21*src[sIdx + dataW * 6 + 2]/4 + 21*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 6];
dst[bIdx +  49  * gap] =  -2*src[sIdx + dataW * 1 + 1] - 2*src[sIdx + dataW * 1 + 2] + 17*src[sIdx + dataW * 1 + 3]/2 + 17*src[sIdx + dataW * 1 + 4]/2 - 2*src[sIdx + dataW * 1 + 5] - 2*src[sIdx + dataW * 1 + 6] + 4*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] - 17*src[sIdx + dataW * 2 + 3] - 17*src[sIdx + dataW * 2 + 4] + 4*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 1]/2 + 5*src[sIdx + dataW * 3 + 2]/2 - 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5]/2 + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1] - 5*src[sIdx + dataW * 4 + 2] + 85*src[sIdx + dataW * 4 + 3]/4 + 85*src[sIdx + dataW * 4 + 4]/4 - 5*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 1]/2 - src[sIdx + dataW * 5 + 2]/2 + 17*src[sIdx + dataW * 5 + 3]/8 + 17*src[sIdx + dataW * 5 + 4]/8 - src[sIdx + dataW * 5 + 5]/2 - src[sIdx + dataW * 5 + 6]/2 + src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] - 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 + src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  50  * gap] =  2*src[sIdx + dataW * 1 + 1] - 2*src[sIdx + dataW * 1 + 2] - 17*src[sIdx + dataW * 1 + 3]/2 + 17*src[sIdx + dataW * 1 + 4]/2 + 2*src[sIdx + dataW * 1 + 5] - 2*src[sIdx + dataW * 1 + 6] - 4*src[sIdx + dataW * 2 + 1] + 4*src[sIdx + dataW * 2 + 2] + 17*src[sIdx + dataW * 2 + 3] - 17*src[sIdx + dataW * 2 + 4] - 4*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 1]/2 + 5*src[sIdx + dataW * 3 + 2]/2 + 85*src[sIdx + dataW * 3 + 3]/8 - 85*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5]/2 + 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1] - 5*src[sIdx + dataW * 4 + 2] - 85*src[sIdx + dataW * 4 + 3]/4 + 85*src[sIdx + dataW * 4 + 4]/4 + 5*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 1]/2 - src[sIdx + dataW * 5 + 2]/2 - 17*src[sIdx + dataW * 5 + 3]/8 + 17*src[sIdx + dataW * 5 + 4]/8 + src[sIdx + dataW * 5 + 5]/2 - src[sIdx + dataW * 5 + 6]/2 - src[sIdx + dataW * 6 + 1] + src[sIdx + dataW * 6 + 2] + 17*src[sIdx + dataW * 6 + 3]/4 - 17*src[sIdx + dataW * 6 + 4]/4 - src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  51  * gap] =  -src[sIdx + dataW * 1 + 1] - src[sIdx + dataW * 1 + 2]/2 + 5*src[sIdx + dataW * 1 + 3] + 5*src[sIdx + dataW * 1 + 4]/2 - 4*src[sIdx + dataW * 1 + 5] - 2*src[sIdx + dataW * 1 + 6] + 2*src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] - 10*src[sIdx + dataW * 2 + 3] - 5*src[sIdx + dataW * 2 + 4] + 8*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 1]/4 + 5*src[sIdx + dataW * 3 + 2]/8 - 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/8 + 5*src[sIdx + dataW * 3 + 5] + 5*src[sIdx + dataW * 3 + 6]/2 - 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2]/4 + 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4]/4 - 10*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 1]/4 - src[sIdx + dataW * 5 + 2]/8 + 5*src[sIdx + dataW * 5 + 3]/4 + 5*src[sIdx + dataW * 5 + 4]/8 - src[sIdx + dataW * 5 + 5] - src[sIdx + dataW * 5 + 6]/2 + src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 + 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  52  * gap] =  src[sIdx + dataW * 1 + 1] - src[sIdx + dataW * 1 + 2]/2 - 5*src[sIdx + dataW * 1 + 3] + 5*src[sIdx + dataW * 1 + 4]/2 + 4*src[sIdx + dataW * 1 + 5] - 2*src[sIdx + dataW * 1 + 6] - 2*src[sIdx + dataW * 2 + 1] + src[sIdx + dataW * 2 + 2] + 10*src[sIdx + dataW * 2 + 3] - 5*src[sIdx + dataW * 2 + 4] - 8*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 1]/4 + 5*src[sIdx + dataW * 3 + 2]/8 + 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/8 - 5*src[sIdx + dataW * 3 + 5] + 5*src[sIdx + dataW * 3 + 6]/2 + 5*src[sIdx + dataW * 4 + 1]/2 - 5*src[sIdx + dataW * 4 + 2]/4 - 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4]/4 + 10*src[sIdx + dataW * 4 + 5] - 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 1]/4 - src[sIdx + dataW * 5 + 2]/8 - 5*src[sIdx + dataW * 5 + 3]/4 + 5*src[sIdx + dataW * 5 + 4]/8 + src[sIdx + dataW * 5 + 5] - src[sIdx + dataW * 5 + 6]/2 - src[sIdx + dataW * 6 + 1]/2 + src[sIdx + dataW * 6 + 2]/4 + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4]/4 - 2*src[sIdx + dataW * 6 + 5] + src[sIdx + dataW * 6 + 6];
dst[bIdx +  53  * gap] =  -4*src[sIdx + dataW * 1 + 1] - 8*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3] + 10*src[sIdx + dataW * 1 + 4] - src[sIdx + dataW * 1 + 5] - 2*src[sIdx + dataW * 1 + 6] + 8*src[sIdx + dataW * 2 + 1] + 16*src[sIdx + dataW * 2 + 2] - 10*src[sIdx + dataW * 2 + 3] - 20*src[sIdx + dataW * 2 + 4] + 2*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] + 5*src[sIdx + dataW * 3 + 1] + 10*src[sIdx + dataW * 3 + 2] - 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/2 + 5*src[sIdx + dataW * 3 + 5]/4 + 5*src[sIdx + dataW * 3 + 6]/2 - 10*src[sIdx + dataW * 4 + 1] - 20*src[sIdx + dataW * 4 + 2] + 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4] - 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6] - src[sIdx + dataW * 5 + 1] - 2*src[sIdx + dataW * 5 + 2] + 5*src[sIdx + dataW * 5 + 3]/4 + 5*src[sIdx + dataW * 5 + 4]/2 - src[sIdx + dataW * 5 + 5]/4 - src[sIdx + dataW * 5 + 6]/2 + 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] - 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] + src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  54  * gap] =  4*src[sIdx + dataW * 1 + 1] - 8*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3] + 10*src[sIdx + dataW * 1 + 4] + src[sIdx + dataW * 1 + 5] - 2*src[sIdx + dataW * 1 + 6] - 8*src[sIdx + dataW * 2 + 1] + 16*src[sIdx + dataW * 2 + 2] + 10*src[sIdx + dataW * 2 + 3] - 20*src[sIdx + dataW * 2 + 4] - 2*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 6] - 5*src[sIdx + dataW * 3 + 1] + 10*src[sIdx + dataW * 3 + 2] + 25*src[sIdx + dataW * 3 + 3]/4 - 25*src[sIdx + dataW * 3 + 4]/2 - 5*src[sIdx + dataW * 3 + 5]/4 + 5*src[sIdx + dataW * 3 + 6]/2 + 10*src[sIdx + dataW * 4 + 1] - 20*src[sIdx + dataW * 4 + 2] - 25*src[sIdx + dataW * 4 + 3]/2 + 25*src[sIdx + dataW * 4 + 4] + 5*src[sIdx + dataW * 4 + 5]/2 - 5*src[sIdx + dataW * 4 + 6] + src[sIdx + dataW * 5 + 1] - 2*src[sIdx + dataW * 5 + 2] - 5*src[sIdx + dataW * 5 + 3]/4 + 5*src[sIdx + dataW * 5 + 4]/2 + src[sIdx + dataW * 5 + 5]/4 - src[sIdx + dataW * 5 + 6]/2 - 2*src[sIdx + dataW * 6 + 1] + 4*src[sIdx + dataW * 6 + 2] + 5*src[sIdx + dataW * 6 + 3]/2 - 5*src[sIdx + dataW * 6 + 4] - src[sIdx + dataW * 6 + 5]/2 + src[sIdx + dataW * 6 + 6];
dst[bIdx +  55  * gap] =  2*src[sIdx + dataW * 1 + 1] - 21*src[sIdx + dataW * 1 + 3]/2 + 21*src[sIdx + dataW * 1 + 5]/2 - 2*src[sIdx + dataW * 1 + 7] - 4*src[sIdx + dataW * 2 + 1] + 21*src[sIdx + dataW * 2 + 3] - 21*src[sIdx + dataW * 2 + 5] + 4*src[sIdx + dataW * 2 + 7] - 5*src[sIdx + dataW * 3 + 1]/2 + 105*src[sIdx + dataW * 3 + 3]/8 - 105*src[sIdx + dataW * 3 + 5]/8 + 5*src[sIdx + dataW * 3 + 7]/2 + 5*src[sIdx + dataW * 4 + 1] - 105*src[sIdx + dataW * 4 + 3]/4 + 105*src[sIdx + dataW * 4 + 5]/4 - 5*src[sIdx + dataW * 4 + 7] + src[sIdx + dataW * 5 + 1]/2 - 21*src[sIdx + dataW * 5 + 3]/8 + 21*src[sIdx + dataW * 5 + 5]/8 - src[sIdx + dataW * 5 + 7]/2 - src[sIdx + dataW * 6 + 1] + 21*src[sIdx + dataW * 6 + 3]/4 - 21*src[sIdx + dataW * 6 + 5]/4 + src[sIdx + dataW * 6 + 7];
;
dst[bIdx +  56  * gap] =  -src[sIdx + dataW * 1 + 0] + 21*src[sIdx + dataW * 1 + 2]/4 - 21*src[sIdx + dataW * 1 + 4]/4 + src[sIdx + dataW * 1 + 6] + 21*src[sIdx + dataW * 3 + 0]/4 - 441*src[sIdx + dataW * 3 + 2]/16 + 441*src[sIdx + dataW * 3 + 4]/16 - 21*src[sIdx + dataW * 3 + 6]/4 - 21*src[sIdx + dataW * 5 + 0]/4 + 441*src[sIdx + dataW * 5 + 2]/16 - 441*src[sIdx + dataW * 5 + 4]/16 + 21*src[sIdx + dataW * 5 + 6]/4 + src[sIdx + dataW * 7 + 0] - 21*src[sIdx + dataW * 7 + 2]/4 + 21*src[sIdx + dataW * 7 + 4]/4 - src[sIdx + dataW * 7 + 6];
dst[bIdx +  57  * gap] =  -src[sIdx + dataW * 1 + 1] - src[sIdx + dataW * 1 + 2] + 17*src[sIdx + dataW * 1 + 3]/4 + 17*src[sIdx + dataW * 1 + 4]/4 - src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] + 21*src[sIdx + dataW * 3 + 1]/4 + 21*src[sIdx + dataW * 3 + 2]/4 - 357*src[sIdx + dataW * 3 + 3]/16 - 357*src[sIdx + dataW * 3 + 4]/16 + 21*src[sIdx + dataW * 3 + 5]/4 + 21*src[sIdx + dataW * 3 + 6]/4 - 21*src[sIdx + dataW * 5 + 1]/4 - 21*src[sIdx + dataW * 5 + 2]/4 + 357*src[sIdx + dataW * 5 + 3]/16 + 357*src[sIdx + dataW * 5 + 4]/16 - 21*src[sIdx + dataW * 5 + 5]/4 - 21*src[sIdx + dataW * 5 + 6]/4 + src[sIdx + dataW * 7 + 1] + src[sIdx + dataW * 7 + 2] - 17*src[sIdx + dataW * 7 + 3]/4 - 17*src[sIdx + dataW * 7 + 4]/4 + src[sIdx + dataW * 7 + 5] + src[sIdx + dataW * 7 + 6];
dst[bIdx +  58  * gap] =  src[sIdx + dataW * 1 + 1] - src[sIdx + dataW * 1 + 2] - 17*src[sIdx + dataW * 1 + 3]/4 + 17*src[sIdx + dataW * 1 + 4]/4 + src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] - 21*src[sIdx + dataW * 3 + 1]/4 + 21*src[sIdx + dataW * 3 + 2]/4 + 357*src[sIdx + dataW * 3 + 3]/16 - 357*src[sIdx + dataW * 3 + 4]/16 - 21*src[sIdx + dataW * 3 + 5]/4 + 21*src[sIdx + dataW * 3 + 6]/4 + 21*src[sIdx + dataW * 5 + 1]/4 - 21*src[sIdx + dataW * 5 + 2]/4 - 357*src[sIdx + dataW * 5 + 3]/16 + 357*src[sIdx + dataW * 5 + 4]/16 + 21*src[sIdx + dataW * 5 + 5]/4 - 21*src[sIdx + dataW * 5 + 6]/4 - src[sIdx + dataW * 7 + 1] + src[sIdx + dataW * 7 + 2] + 17*src[sIdx + dataW * 7 + 3]/4 - 17*src[sIdx + dataW * 7 + 4]/4 - src[sIdx + dataW * 7 + 5] + src[sIdx + dataW * 7 + 6];
dst[bIdx +  59  * gap] =  -src[sIdx + dataW * 1 + 1]/2 - src[sIdx + dataW * 1 + 2]/4 + 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4]/4 - 2*src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] + 21*src[sIdx + dataW * 3 + 1]/8 + 21*src[sIdx + dataW * 3 + 2]/16 - 105*src[sIdx + dataW * 3 + 3]/8 - 105*src[sIdx + dataW * 3 + 4]/16 + 21*src[sIdx + dataW * 3 + 5]/2 + 21*src[sIdx + dataW * 3 + 6]/4 - 21*src[sIdx + dataW * 5 + 1]/8 - 21*src[sIdx + dataW * 5 + 2]/16 + 105*src[sIdx + dataW * 5 + 3]/8 + 105*src[sIdx + dataW * 5 + 4]/16 - 21*src[sIdx + dataW * 5 + 5]/2 - 21*src[sIdx + dataW * 5 + 6]/4 + src[sIdx + dataW * 7 + 1]/2 + src[sIdx + dataW * 7 + 2]/4 - 5*src[sIdx + dataW * 7 + 3]/2 - 5*src[sIdx + dataW * 7 + 4]/4 + 2*src[sIdx + dataW * 7 + 5] + src[sIdx + dataW * 7 + 6];
dst[bIdx +  60  * gap] =  src[sIdx + dataW * 1 + 1]/2 - src[sIdx + dataW * 1 + 2]/4 - 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4]/4 + 2*src[sIdx + dataW * 1 + 5] - src[sIdx + dataW * 1 + 6] - 21*src[sIdx + dataW * 3 + 1]/8 + 21*src[sIdx + dataW * 3 + 2]/16 + 105*src[sIdx + dataW * 3 + 3]/8 - 105*src[sIdx + dataW * 3 + 4]/16 - 21*src[sIdx + dataW * 3 + 5]/2 + 21*src[sIdx + dataW * 3 + 6]/4 + 21*src[sIdx + dataW * 5 + 1]/8 - 21*src[sIdx + dataW * 5 + 2]/16 - 105*src[sIdx + dataW * 5 + 3]/8 + 105*src[sIdx + dataW * 5 + 4]/16 + 21*src[sIdx + dataW * 5 + 5]/2 - 21*src[sIdx + dataW * 5 + 6]/4 - src[sIdx + dataW * 7 + 1]/2 + src[sIdx + dataW * 7 + 2]/4 + 5*src[sIdx + dataW * 7 + 3]/2 - 5*src[sIdx + dataW * 7 + 4]/4 - 2*src[sIdx + dataW * 7 + 5] + src[sIdx + dataW * 7 + 6];
dst[bIdx +  61  * gap] =  -2*src[sIdx + dataW * 1 + 1] - 4*src[sIdx + dataW * 1 + 2] + 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4] - src[sIdx + dataW * 1 + 5]/2 - src[sIdx + dataW * 1 + 6] + 21*src[sIdx + dataW * 3 + 1]/2 + 21*src[sIdx + dataW * 3 + 2] - 105*src[sIdx + dataW * 3 + 3]/8 - 105*src[sIdx + dataW * 3 + 4]/4 + 21*src[sIdx + dataW * 3 + 5]/8 + 21*src[sIdx + dataW * 3 + 6]/4 - 21*src[sIdx + dataW * 5 + 1]/2 - 21*src[sIdx + dataW * 5 + 2] + 105*src[sIdx + dataW * 5 + 3]/8 + 105*src[sIdx + dataW * 5 + 4]/4 - 21*src[sIdx + dataW * 5 + 5]/8 - 21*src[sIdx + dataW * 5 + 6]/4 + 2*src[sIdx + dataW * 7 + 1] + 4*src[sIdx + dataW * 7 + 2] - 5*src[sIdx + dataW * 7 + 3]/2 - 5*src[sIdx + dataW * 7 + 4] + src[sIdx + dataW * 7 + 5]/2 + src[sIdx + dataW * 7 + 6];
dst[bIdx +  62  * gap] =  2*src[sIdx + dataW * 1 + 1] - 4*src[sIdx + dataW * 1 + 2] - 5*src[sIdx + dataW * 1 + 3]/2 + 5*src[sIdx + dataW * 1 + 4] + src[sIdx + dataW * 1 + 5]/2 - src[sIdx + dataW * 1 + 6] - 21*src[sIdx + dataW * 3 + 1]/2 + 21*src[sIdx + dataW * 3 + 2] + 105*src[sIdx + dataW * 3 + 3]/8 - 105*src[sIdx + dataW * 3 + 4]/4 - 21*src[sIdx + dataW * 3 + 5]/8 + 21*src[sIdx + dataW * 3 + 6]/4 + 21*src[sIdx + dataW * 5 + 1]/2 - 21*src[sIdx + dataW * 5 + 2] - 105*src[sIdx + dataW * 5 + 3]/8 + 105*src[sIdx + dataW * 5 + 4]/4 + 21*src[sIdx + dataW * 5 + 5]/8 - 21*src[sIdx + dataW * 5 + 6]/4 - 2*src[sIdx + dataW * 7 + 1] + 4*src[sIdx + dataW * 7 + 2] + 5*src[sIdx + dataW * 7 + 3]/2 - 5*src[sIdx + dataW * 7 + 4] - src[sIdx + dataW * 7 + 5]/2 + src[sIdx + dataW * 7 + 6];
dst[bIdx +  63  * gap] =  src[sIdx + dataW * 1 + 1] - 21*src[sIdx + dataW * 1 + 3]/4 + 21*src[sIdx + dataW * 1 + 5]/4 - src[sIdx + dataW * 1 + 7] - 21*src[sIdx + dataW * 3 + 1]/4 + 441*src[sIdx + dataW * 3 + 3]/16 - 441*src[sIdx + dataW * 3 + 5]/16 + 21*src[sIdx + dataW * 3 + 7]/4 + 21*src[sIdx + dataW * 5 + 1]/4 - 441*src[sIdx + dataW * 5 + 3]/16 + 441*src[sIdx + dataW * 5 + 5]/16 - 21*src[sIdx + dataW * 5 + 7]/4 - src[sIdx + dataW * 7 + 1] + 21*src[sIdx + dataW * 7 + 3]/4 - 21*src[sIdx + dataW * 7 + 5]/4 + src[sIdx + dataW * 7 + 7];

	}
}


template <typename Dtype> 
__global__ void winoSrcAddOpt_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = tileH * tileW * inputs * batchs;

		int batchIdx = idx / (tileH * tileW * inputs);
		int inputIdx = (idx / (tileH * tileW)) % inputs ;

		int yIdx = (idx % (tileH * tileW)) / tileW ;
		int xIdx = idx % tileW;

		int bIdx =  idx % gap;
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;

		float C[16]; 

		//// -- project ---- ///

	}
}

template <typename Dtype> 
__global__ void wino4x4SrcAddOpt_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = tileH * tileW * inputs * batchs;

		int batchIdx = idx / (tileH * tileW * inputs);
		int inputIdx = (idx / (tileH * tileW)) % inputs ;

		int yIdx = (idx % (tileH * tileW)) / tileW ;
		int xIdx = idx % tileW;

		int bIdx =  idx % gap;
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 4 + xIdx * 4;


		//// -- project ---- ///
float a[30];
#pragma unroll
for (int i = 0; i < 30; i++) a[i] = SSRC(i/6, i%6);

dst[bIdx +  0  * gap] =  16*(a[0]-a[2]-a[12]) + 4*(a[4]+a[24]-a[2]-a[12]) + 25*a[14] - 5*(a[16]+a[26]) + a[28];
dst[bIdx +  1  * gap] =  16*(-a[1]-a[2]-a[13]-a[14]) + 4*(a[3]+a[4]+a[13]+a[14]-a[25]-a[26]) - 5*(a[15] + a[16]) + a[27] + a[28] ;
dst[bIdx +  2  * gap] =  16*(a[1]-a[2]-a[13]+a[14]) + 4*(-a[3]+a[4]-a[13]+a[14]+a[25]-a[26]) + 5*(a[15]-a[16])  - a[27] + a[28] ;
dst[bIdx +  3  * gap] =  8*(-a[1]+a[3]+a[13]-a[15]) + 4*(-a[2]+a[4]) + 5*(a[14]-a[16]) - 2*(a[25]+a[27]-a[13]+a[15]) - a[26] + a[28] ;
dst[bIdx +  4  * gap] =  8*(a[1]-a[3]-a[13]+a[15]) + 4*(-a[2] + 4*a[4]) + 5*(a[14]-a[16]) + 2*(a[25]-a[27]-a[13]+a[15]) - a[26] + a[28] ;
dst[bIdx +  5  * gap] =  16*(a[1]-a[3]-a[13]) + 4*(a[5]-a[3]-a[13]+a[25]) + 25*a[15] - 5*(a[17] +a[27]) + a[29] ;

//P6
dst[bIdx +  11  * gap] =  16*(-a[7]+a[9]-a[13]+a[15]) + 4*(-a[17]+a[9]-a[11]+a[15]+a[19]+a[25]-a[21]-a[27]) - a[21]-a[27] + a[23] + a[29] ;
dst[bIdx +  17  * gap] =  16*(a[7]-a[9]-a[13]+a[15]) + 4*(a[11]-a[9]+a[15]-a[17]-a[19]+a[25]+a[21]-a[27]) + a[21]-a[27] - a[23] - a[29] ;
dst[bIdx +  23  * gap] =  8*(-a[7]+a[19]+a[9]-a[21]) + 2*(-a[11]+a[9]-a[21]+a[23]) + 4*(-a[13]+a[25]+a[15]-a[27]) + a[15]-a[27] - a[17] + a[29] ;
dst[bIdx +  29  * gap] =  8*(a[7]-a[9]-a[19]+a[21]) + 2*(a[11]-a[9]+a[21]-a[23]) + 4*(-a[13]+a[25]+a[15]-a[27]) + a[15]-a[27] - a[17] + a[29] ;


dst[bIdx +  7  * gap] =  16*(a[7]+a[8]+a[13]+a[14]) - 4*(a[9]-a[10]-a[15]-a[16]-a[19]-a[20]-a[25]-a[26]) + a[21] + a[22] + a[27] + a[28] ;
dst[bIdx +  8  * gap] =  16*(-a[7]+a[8]-a[13]+a[14]) + 4*(a[9]-a[10]+a[15]-a[16]+a[19]-a[20]+a[25]-a[26]) - a[21] + a[22] - a[27] + a[28] ;
dst[bIdx +  9  * gap] =  8*(a[7]-a[9]+a[13]-a[15]) + 4*(a[8]-a[10]+a[14]-a[16]) + 2*(-a[19]+a[21]-a[25]+a[27]) - a[20]  + a[22]  - a[26] + a[28] ;
dst[bIdx +  10  * gap] =  8*(-a[7]+a[9]-a[13]+a[15]) + 4*(a[8]-a[10]+a[14]-a[16]) + 2*(a[19]-a[21]+a[25]-a[27]) - a[20] + a[22] - a[26] + a[28] ;


dst[bIdx +  13  * gap] =  16*(-a[7]-a[8]+a[13]+a[14]) + 4*(a[9]+a[10]-a[15]-a[16]+a[19]+a[20]-a[25]-a[26]) - a[21] - a[22] + a[27] + a[28] ;
dst[bIdx +  14  * gap] =  16*(a[7]-a[8]-a[13]+a[14]) -4*(a[9]+a[10]+a[15]-a[16]-a[19]+a[20]+a[25]-a[26]) + a[21] - a[22] - a[27] + a[28] ;
dst[bIdx +  15  * gap] =  8*(-a[7]+a[9]+a[13]-a[15]) + 4*(-a[8]+a[10]+a[14]-a[16]) + 2*(a[19]-a[21]+a[25]+a[27]) + a[20] - a[22] - a[26] + a[28] ;
dst[bIdx +  16  * gap] =  8*(a[7]-a[9]-a[13]+a[15]) + 4*(a[8]+a[10]+a[14]-a[16]) + 2*(-a[19]+a[21]+a[25]-a[27]) + a[20] - a[22] - a[26] + a[28] ;

dst[bIdx +  19  * gap] =  8*(a[7]+a[8]-a[19]-a[20]) + 2*(-a[9]-a[10]+a[21]+a[22]) + 4*(a[13]+a[14]-a[25]-a[26]) - a[15] - a[16] + a[27] + a[28] ;
dst[bIdx +  20  * gap] =  8*(-a[7]+a[8]+a[19]-a[20]) + 2*(a[9]-a[10]-a[21]+a[22]) + 4*(-a[13]+a[14]+a[25]-a[26]) + a[15] - a[16] - a[27] + a[28] ;
dst[bIdx +  21  * gap] =  4*(a[7]-a[9]-a[19]+a[21]) + 2*(a[8]-a[10]+a[13]-a[15]-a[20]+a[22]-a[25]+a[27]) + a[14] - a[16] - a[26] + a[28] ;
dst[bIdx +  22  * gap] =  4*(-a[7]+a[9]+a[19]-a[21]) + 2*(a[8]-a[10]-a[13]+a[15]-a[20]+a[22]+a[25]-a[27]) + a[14] - a[16] - a[26] + a[28] ;

dst[bIdx +  25  * gap] =  8*(-a[7]-a[8]+a[19]+a[20]) + 2*(a[9]+a[10]-a[21]-a[22]) + 4*(a[13]+a[14]-a[25]-a[26]) - a[15] - a[16] + a[27] + a[28] ;
dst[bIdx +  26  * gap] =  8*(a[7]-a[8]-a[19]+a[20]) + 2*(-a[9]+a[10]+a[21]-a[22]) + 4*(-a[13]+a[14]+a[25]-a[26]) + a[15] - a[16] - a[27] + a[28] ;
dst[bIdx +  27  * gap] =  4*(-a[7]+a[9]+a[19]-a[21]) + 2*(-a[8]+a[10]+a[13]-a[15]+a[20]-a[22]-a[25]+a[27]) + a[14] - a[16] - a[26] + a[28] ;
dst[bIdx +  28  * gap] =  4*(a[7]-a[9]-a[19]+a[21]) + 2*(-a[8]+a[10]-a[13]+a[15]+a[20]-a[22]+a[25]-a[27]) + a[14] - a[16] - a[26] + a[28] ;

//P4
dst[bIdx +  6  * gap] =  16*(-a[6]+a[8]-a[12]+a[14]) + 4*(-a[10]+a[8]+a[14]-a[16]+a[18]+a[24]) - 5*(a[20]+a[26]) + a[22] + a[28] ;
dst[bIdx +  12  * gap] =  16*(a[6]-a[8]-a[12]+a[14]) + 4*(a[10]-a[8]+a[14]-a[16]-a[18]+a[24]) + 5*(a[20]-a[26]) - a[22] + a[28] ;
dst[bIdx +  18  * gap] =  8*(-a[6]+a[8]+a[18]-a[20]) + 2*(-a[10]+a[8]-a[20]+a[22]) + 4*(-a[12]+a[24]) + 5*(a[14]-a[26]) - a[16] + a[28] ;
dst[bIdx +  24  * gap] =  8*(a[6]-a[8]-a[18]+a[20]) +  2*(a[10]-a[8]+a[20]-a[22]) + 4*(-a[12]+a[24]) + 5*(a[14]-a[26]) - a[16] + a[28] ;

#pragma unroll
for (int i = 0;i<6;i++) a[i] = SSRC(5,i);

dst[bIdx +  30  * gap] =  16*(a[6]-a[8]-a[18]) + 4*(a[10]-a[8]-a[18]+a[0]-a[22]-a[2])  + 25*a[20] - a[22] - a[2] + a[4] ;
dst[bIdx +  31  * gap] =  16*(-a[7]-a[8]+a[19]+a[20]) + 4*(a[9]+a[10]+a[19]+a[20]-a[21]-a[22]-a[1]-a[2]) -a[21]-a[22] + a[3] + a[4] ;
dst[bIdx +  32  * gap] =  16*(a[7]-a[8]-a[19]+a[20]) + 4*(-a[9]+a[10]-a[19]+a[20]+a[21]-a[22]+a[1]-a[2]) + a[21] - a[22] - a[3] + a[4] ;
dst[bIdx +  33  * gap] =  8*(-a[7]+a[9]+a[19]-a[21]) + 4*(-a[8]+a[10]+a[20]-a[22]) +a[20] - a[22] + 2*(-a[1]+a[19]-a[21]+a[3]) - a[2] + a[4] ;
dst[bIdx +  34  * gap] =  8*(a[7]-a[9]-a[19]+a[21]) + 4*(-a[8]+a[10]+a[20]-a[22]) + a[20] - a[22] + 2*(a[1]-a[19]+a[21]-a[3]) - a[2] + a[4] ;
dst[bIdx +  35  * gap] =  16*(a[7]-a[9]-a[19]) + 4*(a[11]-a[9]-a[19]+a[1]) + 25*a[21] - 5*(a[23]+a[3]) + a[5] ;	
	}
}

template <typename Dtype> 
__global__ void wino6x6SrcAddOpt_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = tileH * tileW * inputs * batchs;

		int batchIdx = idx / (tileH * tileW * inputs);
		int inputIdx = (idx / (tileH * tileW)) % inputs ;

		int yIdx = (idx % (tileH * tileW)) / tileW ;
		int xIdx = idx % tileW;

		int bIdx =  idx % gap;
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;


		//// -- project ---- ///

	}
}



template <typename Dtype> 
__global__ void winoMulti_gpu_kernel(const Dtype *u_matrix, const Dtype *v_matrix, Dtype *m_matrix,
                                    const int Ah, const int Bw, const int Aw, const float alpha, const float beta)
{
   const Dtype *A = u_matrix + blockIdx.z * Ah * Aw;
   const Dtype *B = v_matrix + blockIdx.z * Aw * Bw;
   Dtype *C = m_matrix + blockIdx.z * Ah * Bw;

   int col = blockIdx.x * NUM_THREADS + threadIdx.x;
   int row = blockIdx.y * NUM_THREADS + threadIdx.y;
   __shared__ float As[NUM_THREADS][NUM_THREADS];
   __shared__ float Bs[NUM_THREADS][NUM_THREADS];

   float acc = 0.0;
   for (int i = 0; i <= (Aw-1) / NUM_THREADS+1; i++) {
     int tr = row;
     int tc = i * NUM_THREADS + threadIdx.x;
     if (tr < Ah && tc < Aw)
         As[threadIdx.y][threadIdx.x] = A[row * Aw + i * NUM_THREADS + threadIdx.x];
     else
         As[threadIdx.y][threadIdx.x] = 0.0;
     tr = i * NUM_THREADS + threadIdx.y;
     tc = col;
     if (tr < Aw && tc < Bw)
       Bs[threadIdx.y][threadIdx.x] = B[(i * NUM_THREADS + threadIdx.y) * Bw + col];
     else
       Bs[threadIdx.y][threadIdx.x] = 0.0;
     __syncthreads();
 //#pragma unroll
     for (int k = 0; k < NUM_THREADS && i * NUM_THREADS + k < Aw; k++)
       acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
     __syncthreads();
   }
   if (row >= Ah || col >= Bw) return;
   C[row * Bw + col] = acc;
  /*
	int bx = blockIdx.x;
	int by = blockIdx.y; 
	int bz = blockIdx.z; 

	int tx = threadIdx.x; 
	int ty = threadIdx.y; 

	int aBegin = bz * Aw * Ah + Aw *32 * by; 
	int aEnd = Aw; 
	int aStep = 32; 

	int bBegin = bz * Bw * Aw + 32 * bx; 
	int bStep = 32; 

	float Csub = 0; 

	for(int a = 0, b = 0; a < aEnd; a += aStep, b+= bStep)
	{
		__shared__ float As[32][32]; 
		__shared__ float Bs[32][32]; 

		if( ((tx+a) < Aw) && ((32 * by + ty) < Ah))
			As[ty][tx] = A[aBegin + a + Aw * ty + tx]; 
		else 
			As[ty][tx] = 0; 

		if( ((32 * bx + tx) < Bw) && ( (b + ty) < Aw))
			Bs[ty][tx] = B[bBegin + Bw * (b + ty) + tx]; 
		else 
			Bs[ty][tx] = 0; 

		__syncthreads(); 

#pragma unroll
		for(int k = 0; k < 32; k++)
		{
			Csub += As[ty][k] * Bs[k][tx]; 
		}

		__syncthreads(); 
	}

	int cW = 32 * bx + tx; 
	int cH = 32 * by + ty;

	if((cW < Bw) && (cH < Ah))   
		C[bz * Bw * Ah + Bw * cH + cW] = Csub; 
*/
}


template <typename Dtype> 
__global__ void winoDst_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

		float tmp ; 
		tmp = src[mIdx + gap * 0] + src[mIdx + gap * 1] + src[mIdx + gap * 2] + src[mIdx + gap * 4] + src[mIdx + gap * 5] + src[mIdx + gap * 6] + src[mIdx + gap * 8] + src[mIdx + gap * 9] + src[mIdx + gap * 10];
//		tmp = fabs(tmp) < 0.000001 ? 0 : tmp; 
//      dst[rIdx + 0] = bias[outIdx] + tmp; 
		dst[rIdx + 0] = tmp; 
               
      tmp = src[mIdx + gap * 1] - src[mIdx + gap * 2] - src[mIdx + gap * 3] + src[mIdx + gap * 5] - src[mIdx + gap * 6] - src[mIdx + gap * 7] + src[mIdx + gap * 9] - src[mIdx + gap * 10] - src[mIdx + gap * 11];
//		tmp = fabs(tmp) < 0.000001? 0 : tmp; 
//		dst[rIdx + 1] = bias[utIdx] + tmp;
		dst[rIdx + 1] = tmp;

		tmp = src[mIdx + gap *4] + src[mIdx + gap * 5] + src[mIdx + gap * 6] - src[mIdx + gap * 8] - src[mIdx + gap * 9] - src[mIdx + gap * 10] - src[mIdx + gap * 12] - src[mIdx + gap * 13] - src[mIdx + gap * 14];
//		tmp = fabs(tmp) < 0.00000 ? 0 : tmp; 
//		dst[rIdx + outW] = bias[outIdx] + tmp; 
		dst[rIdx + outW] =  tmp; 

		tmp = src[mIdx + gap * 5] - src[mIdx + gap * 6] - src[mIdx + gap * 7] - src[mIdx + gap * 9] + src[mIdx + gap * 10] + src[mIdx + gap * 11] - src[mIdx + gap * 13] + src[mIdx + gap * 14] + src[mIdx + gap * 15];
//		tmp = fabs(tmp) < 0.000001 ? 0 : tmp; 
//		dst[rIdx + outW + 1] = bias[outIdx] + tmp; 
		dst[rIdx + outW + 1] = tmp; 
/*
    if (threadIdx.x + blockIdx.x + threadIdx.y + blockIdx.y == 0) {
       printf("original dst:");
       for (int i = 0; i < 16; i++) {
         if (i % 4 == 0) printf("\n");
         printf("%f ", src[mIdx + i * gap]);
       }
       printf("\n\n");
       printf("Dst:\n");
       for (int i = 0; i < 2; i++) {
         for (int j = 0; j < 2; j++) {
           printf("%f ", dst[rIdx + j + i * outW]);
         }
         printf("\n");
       }
     }
*/
	}
}

template <typename Dtype> 
__global__ void wino4x4Dst_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 4 + xIdx * 4;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;
			
						
		//// -- project ---- //		
    dst[rIdx] =  src[mIdx + gap * 0] + src[mIdx + gap * 1] + src[mIdx + gap * 10] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + src[mIdx + gap * 15] + src[mIdx + gap * 16] + src[mIdx + gap * 18] + src[mIdx + gap * 19] + src[mIdx + gap * 2] + src[mIdx + gap * 20] + src[mIdx + gap * 21] + src[mIdx + gap * 22] + src[mIdx + gap * 24] + src[mIdx + gap * 25] + src[mIdx + gap * 26] + src[mIdx + gap * 27] + src[mIdx + gap * 28] + src[mIdx + gap * 3] + src[mIdx + gap * 4] + src[mIdx + gap * 6] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + src[mIdx + gap * 9];
    dst[rIdx + 1] =  src[mIdx + gap * 1] - 2*src[mIdx + gap * 10] + src[mIdx + gap * 13] - src[mIdx + gap * 14] + 2*src[mIdx + gap * 15] - 2*src[mIdx + gap * 16] + src[mIdx + gap * 19] - src[mIdx + gap * 2] - src[mIdx + gap * 20] + 2*src[mIdx + gap * 21] - 2*src[mIdx + gap * 22] + src[mIdx + gap * 25] - src[mIdx + gap * 26] + 2*src[mIdx + gap * 27] - 2*src[mIdx + gap * 28] + 2*src[mIdx + gap * 3] - 2*src[mIdx + gap * 4] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 2*src[mIdx + gap * 9];
    dst[rIdx + 2] =  src[mIdx + gap * 1] + 4*src[mIdx + gap * 10] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + 4*src[mIdx + gap * 15] + 4*src[mIdx + gap * 16] + src[mIdx + gap * 19] + src[mIdx + gap * 2] + src[mIdx + gap * 20] + 4*src[mIdx + gap * 21] + 4*src[mIdx + gap * 22] + src[mIdx + gap * 25] + src[mIdx + gap * 26] + 4*src[mIdx + gap * 27] + 4*src[mIdx + gap * 28] + 4*src[mIdx + gap * 3] + 4*src[mIdx + gap * 4] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + 4*src[mIdx + gap * 9];
    dst[rIdx + 3] =  src[mIdx + gap * 1] - 8*src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 13] - src[mIdx + gap * 14] + 8*src[mIdx + gap * 15] - 8*src[mIdx + gap * 16] + src[mIdx + gap * 17] + src[mIdx + gap * 19] - src[mIdx + gap * 2] - src[mIdx + gap * 20] + 8*src[mIdx + gap * 21] - 8*src[mIdx + gap * 22] + src[mIdx + gap * 23] + src[mIdx + gap * 25] - src[mIdx + gap * 26] + 8*src[mIdx + gap * 27] - 8*src[mIdx + gap * 28] + src[mIdx + gap * 29] + 8*src[mIdx + gap * 3] - 8*src[mIdx + gap * 4] + src[mIdx + gap * 5] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 8*src[mIdx + gap * 9];

    dst[rIdx + outW] =  src[mIdx + gap * 10] - src[mIdx + gap * 12] - src[mIdx + gap * 13] - src[mIdx + gap * 14] - src[mIdx + gap * 15] - src[mIdx + gap * 16] + 2*src[mIdx + gap * 18] + 2*src[mIdx + gap * 19] + 2*src[mIdx + gap * 20] + 2*src[mIdx + gap * 21] + 2*src[mIdx + gap * 22] - 2*src[mIdx + gap * 24] - 2*src[mIdx + gap * 25] - 2*src[mIdx + gap * 26] - 2*src[mIdx + gap * 27] - 2*src[mIdx + gap * 28] + src[mIdx + gap * 6] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + src[mIdx + gap * 9];
    dst[rIdx + 1 + outW] =  -2*src[mIdx + gap * 10] - src[mIdx + gap * 13] + src[mIdx + gap * 14] - 2*src[mIdx + gap * 15] + 2*src[mIdx + gap * 16] + 2*src[mIdx + gap * 19] - 2*src[mIdx + gap * 20] + 4*src[mIdx + gap * 21] - 4*src[mIdx + gap * 22] - 2*src[mIdx + gap * 25] + 2*src[mIdx + gap * 26] - 4*src[mIdx + gap * 27] + 4*src[mIdx + gap * 28] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 2*src[mIdx + gap * 9];
    dst[rIdx + 2 + outW] =  4*src[mIdx + gap * 10] - src[mIdx + gap * 13] - src[mIdx + gap * 14] - 4*src[mIdx + gap * 15] - 4*src[mIdx + gap * 16] + 2*src[mIdx + gap * 19] + 2*src[mIdx + gap * 20] + 8*src[mIdx + gap * 21] + 8*src[mIdx + gap * 22] - 2*src[mIdx + gap * 25] - 2*src[mIdx + gap * 26] - 8*src[mIdx + gap * 27] - 8*src[mIdx + gap * 28] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + 4*src[mIdx + gap * 9];
    dst[rIdx + 3 + outW] =  -8*src[mIdx + gap * 10] + src[mIdx + gap * 11] - src[mIdx + gap * 13] + src[mIdx + gap * 14] - 8*src[mIdx + gap * 15] + 8*src[mIdx + gap * 16] - src[mIdx + gap * 17] + 2*src[mIdx + gap * 19] - 2*src[mIdx + gap * 20] + 16*src[mIdx + gap * 21] - 16*src[mIdx + gap * 22] + 2*src[mIdx + gap * 23] - 2*src[mIdx + gap * 25] + 2*src[mIdx + gap * 26] - 16*src[mIdx + gap * 27] + 16*src[mIdx + gap * 28] - 2*src[mIdx + gap * 29] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 8*src[mIdx + gap * 9];

    dst[rIdx + 2 * outW] =  src[mIdx + gap * 10] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + src[mIdx + gap * 15] + src[mIdx + gap * 16] + 4*src[mIdx + gap * 18] + 4*src[mIdx + gap * 19] + 4*src[mIdx + gap * 20] + 4*src[mIdx + gap * 21] + 4*src[mIdx + gap * 22] + 4*src[mIdx + gap * 24] + 4*src[mIdx + gap * 25] + 4*src[mIdx + gap * 26] + 4*src[mIdx + gap * 27] + 4*src[mIdx + gap * 28] + src[mIdx + gap * 6] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + src[mIdx + gap * 9];
    dst[rIdx + 1 + 2 * outW] =  -2*src[mIdx + gap * 10] + src[mIdx + gap * 13] - src[mIdx + gap * 14] + 2*src[mIdx + gap * 15] - 2*src[mIdx + gap * 16] + 4*src[mIdx + gap * 19] - 4*src[mIdx + gap * 20] + 8*src[mIdx + gap * 21] - 8*src[mIdx + gap * 22] + 4*src[mIdx + gap * 25] - 4*src[mIdx + gap * 26] + 8*src[mIdx + gap * 27] - 8*src[mIdx + gap * 28] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 2*src[mIdx + gap * 9];
    dst[rIdx + 2 + 2 * outW] =  4*src[mIdx + gap * 10] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + 4*src[mIdx + gap * 15] + 4*src[mIdx + gap * 16] + 4*src[mIdx + gap * 19] + 4*src[mIdx + gap * 20] + 16*src[mIdx + gap * 21] + 16*src[mIdx + gap * 22] + 4*src[mIdx + gap * 25] + 4*src[mIdx + gap * 26] + 16*src[mIdx + gap * 27] + 16*src[mIdx + gap * 28] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + 4*src[mIdx + gap * 9];
    dst[rIdx + 3 + 2 * outW] =  -8*src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 13] - src[mIdx + gap * 14] + 8*src[mIdx + gap * 15] - 8*src[mIdx + gap * 16] + src[mIdx + gap * 17] + 4*src[mIdx + gap * 19] - 4*src[mIdx + gap * 20] + 32*src[mIdx + gap * 21] - 32*src[mIdx + gap * 22] + 4*src[mIdx + gap * 23] + 4*src[mIdx + gap * 25] - 4*src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] - 32*src[mIdx + gap * 28] + 4*src[mIdx + gap * 29] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 8*src[mIdx + gap * 9];

    dst[rIdx + 3 * outW] =  src[mIdx + gap * 10] - src[mIdx + gap * 12] - src[mIdx + gap * 13] - src[mIdx + gap * 14] - src[mIdx + gap * 15] - src[mIdx + gap * 16] + 8*src[mIdx + gap * 18] + 8*src[mIdx + gap * 19] + 8*src[mIdx + gap * 20] + 8*src[mIdx + gap * 21] + 8*src[mIdx + gap * 22] - 8*src[mIdx + gap * 24] - 8*src[mIdx + gap * 25] - 8*src[mIdx + gap * 26] - 8*src[mIdx + gap * 27] - 8*src[mIdx + gap * 28] + src[mIdx + gap * 30] + src[mIdx + gap * 31] + src[mIdx + gap * 32] + src[mIdx + gap * 33] + src[mIdx + gap * 34] + src[mIdx + gap * 6] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + src[mIdx + gap * 9];
    dst[rIdx + 1 + 3 * outW] =  -2*src[mIdx + gap * 10] - src[mIdx + gap * 13] + src[mIdx + gap * 14] - 2*src[mIdx + gap * 15] + 2*src[mIdx + gap * 16] + 8*src[mIdx + gap * 19] - 8*src[mIdx + gap * 20] + 16*src[mIdx + gap * 21] - 16*src[mIdx + gap * 22] - 8*src[mIdx + gap * 25] + 8*src[mIdx + gap * 26] - 16*src[mIdx + gap * 27] + 16*src[mIdx + gap * 28] + src[mIdx + gap * 31] - src[mIdx + gap * 32] + 2*src[mIdx + gap * 33] - 2*src[mIdx + gap * 34] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 2*src[mIdx + gap * 9];
    dst[rIdx + 2 + 3 * outW] =  4*src[mIdx + gap * 10] - src[mIdx + gap * 13] - src[mIdx + gap * 14] - 4*src[mIdx + gap * 15] - 4*src[mIdx + gap * 16] + 8*src[mIdx + gap * 19] + 8*src[mIdx + gap * 20] + 32*src[mIdx + gap * 21] + 32*src[mIdx + gap * 22] - 8*src[mIdx + gap * 25] - 8*src[mIdx + gap * 26] - 32*src[mIdx + gap * 27] - 32*src[mIdx + gap * 28] + src[mIdx + gap * 31] + src[mIdx + gap * 32] + 4*src[mIdx + gap * 33] + 4*src[mIdx + gap * 34] + src[mIdx + gap * 7] + src[mIdx + gap * 8] + 4*src[mIdx + gap * 9];
    dst[rIdx + 3 + 3 * outW] =  -8*src[mIdx + gap * 10] + src[mIdx + gap * 11] - src[mIdx + gap * 13] + src[mIdx + gap * 14] - 8*src[mIdx + gap * 15] + 8*src[mIdx + gap * 16] - src[mIdx + gap * 17] + 8*src[mIdx + gap * 19] - 8*src[mIdx + gap * 20] + 64*src[mIdx + gap * 21] - 64*src[mIdx + gap * 22] + 8*src[mIdx + gap * 23] - 8*src[mIdx + gap * 25] + 8*src[mIdx + gap * 26] - 64*src[mIdx + gap * 27] + 64*src[mIdx + gap * 28] - 8*src[mIdx + gap * 29] + src[mIdx + gap * 31] - src[mIdx + gap * 32] + 8*src[mIdx + gap * 33] - 8*src[mIdx + gap * 34] + src[mIdx + gap * 35] + src[mIdx + gap * 7] - src[mIdx + gap * 8] + 8*src[mIdx + gap * 9];
/*
    if (threadIdx.x + blockIdx.x + threadIdx.y + blockIdx.y == 0) {
      printf("original dst:");
      for (int i = 0; i < 36; i++) {
        if (i % 6 == 0) printf("\n");
        printf("%f ", src[mIdx + i * gap]);
      }
      printf("\n\n");
      printf("Dst:\n");
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          printf("%f ", dst[rIdx + j + i * outW]);
        }
        printf("\n");
      }
    }
  */
  }
}

template <typename Dtype> 
__global__ void wino6x6Dst_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 6 + xIdx * 6;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;
			
						
		//// -- project ---- //		

dst[rIdx + outW * 0 + 0]  =  src[mIdx + gap * 0] + src[mIdx + gap * 1] + src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + src[mIdx + gap * 16] + src[mIdx + gap * 17] + src[mIdx + gap * 18] + src[mIdx + gap * 19] + src[mIdx + gap * 2] + src[mIdx + gap * 20] + src[mIdx + gap * 21] + src[mIdx + gap * 22] + src[mIdx + gap * 24] + src[mIdx + gap * 25] + src[mIdx + gap * 26] + src[mIdx + gap * 27] + src[mIdx + gap * 28] + src[mIdx + gap * 29] + src[mIdx + gap * 3] + src[mIdx + gap * 30] + src[mIdx + gap * 32] + src[mIdx + gap * 33] + src[mIdx + gap * 34] + src[mIdx + gap * 35] + src[mIdx + gap * 36] + src[mIdx + gap * 37] + src[mIdx + gap * 38] + src[mIdx + gap * 4] + src[mIdx + gap * 40] + src[mIdx + gap * 41] + src[mIdx + gap * 42] + src[mIdx + gap * 43] + src[mIdx + gap * 44] + src[mIdx + gap * 45] + src[mIdx + gap * 46] + src[mIdx + gap * 48] + src[mIdx + gap * 49] + src[mIdx + gap * 5] + src[mIdx + gap * 50] + src[mIdx + gap * 51] + src[mIdx + gap * 52] + src[mIdx + gap * 53] + src[mIdx + gap * 54] + src[mIdx + gap * 6] + src[mIdx + gap * 8] + src[mIdx + gap * 9];
dst[rIdx + outW * 0 + 1]  =  src[mIdx + gap * 1] - src[mIdx + gap * 10] + 2*src[mIdx + gap * 11] - 2*src[mIdx + gap * 12] + src[mIdx + gap * 13]/2 - src[mIdx + gap * 14]/2 + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 2*src[mIdx + gap * 19] - src[mIdx + gap * 2] - 2*src[mIdx + gap * 20] + src[mIdx + gap * 21]/2 - src[mIdx + gap * 22]/2 + src[mIdx + gap * 25] - src[mIdx + gap * 26] + 2*src[mIdx + gap * 27] - 2*src[mIdx + gap * 28] + src[mIdx + gap * 29]/2 + 2*src[mIdx + gap * 3] - src[mIdx + gap * 30]/2 + src[mIdx + gap * 33] - src[mIdx + gap * 34] + 2*src[mIdx + gap * 35] - 2*src[mIdx + gap * 36] + src[mIdx + gap * 37]/2 - src[mIdx + gap * 38]/2 - 2*src[mIdx + gap * 4] + src[mIdx + gap * 41] - src[mIdx + gap * 42] + 2*src[mIdx + gap * 43] - 2*src[mIdx + gap * 44] + src[mIdx + gap * 45]/2 - src[mIdx + gap * 46]/2 + src[mIdx + gap * 49] + src[mIdx + gap * 5]/2 - src[mIdx + gap * 50] + 2*src[mIdx + gap * 51] - 2*src[mIdx + gap * 52] + src[mIdx + gap * 53]/2 - src[mIdx + gap * 54]/2 - src[mIdx + gap * 6]/2 + src[mIdx + gap * 9];
dst[rIdx + outW * 0 + 2]  =  src[mIdx + gap * 1] + src[mIdx + gap * 10] + 4*src[mIdx + gap * 11] + 4*src[mIdx + gap * 12] + src[mIdx + gap * 13]/4 + src[mIdx + gap * 14]/4 + src[mIdx + gap * 17] + src[mIdx + gap * 18] + 4*src[mIdx + gap * 19] + src[mIdx + gap * 2] + 4*src[mIdx + gap * 20] + src[mIdx + gap * 21]/4 + src[mIdx + gap * 22]/4 + src[mIdx + gap * 25] + src[mIdx + gap * 26] + 4*src[mIdx + gap * 27] + 4*src[mIdx + gap * 28] + src[mIdx + gap * 29]/4 + 4*src[mIdx + gap * 3] + src[mIdx + gap * 30]/4 + src[mIdx + gap * 33] + src[mIdx + gap * 34] + 4*src[mIdx + gap * 35] + 4*src[mIdx + gap * 36] + src[mIdx + gap * 37]/4 + src[mIdx + gap * 38]/4 + 4*src[mIdx + gap * 4] + src[mIdx + gap * 41] + src[mIdx + gap * 42] + 4*src[mIdx + gap * 43] + 4*src[mIdx + gap * 44] + src[mIdx + gap * 45]/4 + src[mIdx + gap * 46]/4 + src[mIdx + gap * 49] + src[mIdx + gap * 5]/4 + src[mIdx + gap * 50] + 4*src[mIdx + gap * 51] + 4*src[mIdx + gap * 52] + src[mIdx + gap * 53]/4 + src[mIdx + gap * 54]/4 + src[mIdx + gap * 6]/4 + src[mIdx + gap * 9];
dst[rIdx + outW * 0 + 3]  =  src[mIdx + gap * 1] - src[mIdx + gap * 10] + 8*src[mIdx + gap * 11] - 8*src[mIdx + gap * 12] + src[mIdx + gap * 13]/8 - src[mIdx + gap * 14]/8 + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 8*src[mIdx + gap * 19] - src[mIdx + gap * 2] - 8*src[mIdx + gap * 20] + src[mIdx + gap * 21]/8 - src[mIdx + gap * 22]/8 + src[mIdx + gap * 25] - src[mIdx + gap * 26] + 8*src[mIdx + gap * 27] - 8*src[mIdx + gap * 28] + src[mIdx + gap * 29]/8 + 8*src[mIdx + gap * 3] - src[mIdx + gap * 30]/8 + src[mIdx + gap * 33] - src[mIdx + gap * 34] + 8*src[mIdx + gap * 35] - 8*src[mIdx + gap * 36] + src[mIdx + gap * 37]/8 - src[mIdx + gap * 38]/8 - 8*src[mIdx + gap * 4] + src[mIdx + gap * 41] - src[mIdx + gap * 42] + 8*src[mIdx + gap * 43] - 8*src[mIdx + gap * 44] + src[mIdx + gap * 45]/8 - src[mIdx + gap * 46]/8 + src[mIdx + gap * 49] + src[mIdx + gap * 5]/8 - src[mIdx + gap * 50] + 8*src[mIdx + gap * 51] - 8*src[mIdx + gap * 52] + src[mIdx + gap * 53]/8 - src[mIdx + gap * 54]/8 - src[mIdx + gap * 6]/8 + src[mIdx + gap * 9];
dst[rIdx + outW * 0 + 4]  =  src[mIdx + gap * 1] + src[mIdx + gap * 10] + 16*src[mIdx + gap * 11] + 16*src[mIdx + gap * 12] + src[mIdx + gap * 13]/16 + src[mIdx + gap * 14]/16 + src[mIdx + gap * 17] + src[mIdx + gap * 18] + 16*src[mIdx + gap * 19] + src[mIdx + gap * 2] + 16*src[mIdx + gap * 20] + src[mIdx + gap * 21]/16 + src[mIdx + gap * 22]/16 + src[mIdx + gap * 25] + src[mIdx + gap * 26] + 16*src[mIdx + gap * 27] + 16*src[mIdx + gap * 28] + src[mIdx + gap * 29]/16 + 16*src[mIdx + gap * 3] + src[mIdx + gap * 30]/16 + src[mIdx + gap * 33] + src[mIdx + gap * 34] + 16*src[mIdx + gap * 35] + 16*src[mIdx + gap * 36] + src[mIdx + gap * 37]/16 + src[mIdx + gap * 38]/16 + 16*src[mIdx + gap * 4] + src[mIdx + gap * 41] + src[mIdx + gap * 42] + 16*src[mIdx + gap * 43] + 16*src[mIdx + gap * 44] + src[mIdx + gap * 45]/16 + src[mIdx + gap * 46]/16 + src[mIdx + gap * 49] + src[mIdx + gap * 5]/16 + src[mIdx + gap * 50] + 16*src[mIdx + gap * 51] + 16*src[mIdx + gap * 52] + src[mIdx + gap * 53]/16 + src[mIdx + gap * 54]/16 + src[mIdx + gap * 6]/16 + src[mIdx + gap * 9];
dst[rIdx + outW * 0 + 5]  =  src[mIdx + gap * 1] - src[mIdx + gap * 10] + 32*src[mIdx + gap * 11] - 32*src[mIdx + gap * 12] + src[mIdx + gap * 13]/32 - src[mIdx + gap * 14]/32 + src[mIdx + gap * 15] + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 32*src[mIdx + gap * 19] - src[mIdx + gap * 2] - 32*src[mIdx + gap * 20] + src[mIdx + gap * 21]/32 - src[mIdx + gap * 22]/32 + src[mIdx + gap * 23] + src[mIdx + gap * 25] - src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] - 32*src[mIdx + gap * 28] + src[mIdx + gap * 29]/32 + 32*src[mIdx + gap * 3] - src[mIdx + gap * 30]/32 + src[mIdx + gap * 31] + src[mIdx + gap * 33] - src[mIdx + gap * 34] + 32*src[mIdx + gap * 35] - 32*src[mIdx + gap * 36] + src[mIdx + gap * 37]/32 - src[mIdx + gap * 38]/32 + src[mIdx + gap * 39] - 32*src[mIdx + gap * 4] + src[mIdx + gap * 41] - src[mIdx + gap * 42] + 32*src[mIdx + gap * 43] - 32*src[mIdx + gap * 44] + src[mIdx + gap * 45]/32 - src[mIdx + gap * 46]/32 + src[mIdx + gap * 47] + src[mIdx + gap * 49] + src[mIdx + gap * 5]/32 - src[mIdx + gap * 50] + 32*src[mIdx + gap * 51] - 32*src[mIdx + gap * 52] + src[mIdx + gap * 53]/32 - src[mIdx + gap * 54]/32 + src[mIdx + gap * 55] - src[mIdx + gap * 6]/32 + src[mIdx + gap * 7] + src[mIdx + gap * 9];

dst[rIdx + outW * 1 + 0]  =  src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] - src[mIdx + gap * 16] - src[mIdx + gap * 17] - src[mIdx + gap * 18] - src[mIdx + gap * 19] - src[mIdx + gap * 20] - src[mIdx + gap * 21] - src[mIdx + gap * 22] + 2*src[mIdx + gap * 24] + 2*src[mIdx + gap * 25] + 2*src[mIdx + gap * 26] + 2*src[mIdx + gap * 27] + 2*src[mIdx + gap * 28] + 2*src[mIdx + gap * 29] + 2*src[mIdx + gap * 30] - 2*src[mIdx + gap * 32] - 2*src[mIdx + gap * 33] - 2*src[mIdx + gap * 34] - 2*src[mIdx + gap * 35] - 2*src[mIdx + gap * 36] - 2*src[mIdx + gap * 37] - 2*src[mIdx + gap * 38] + src[mIdx + gap * 40]/2 + src[mIdx + gap * 41]/2 + src[mIdx + gap * 42]/2 + src[mIdx + gap * 43]/2 + src[mIdx + gap * 44]/2 + src[mIdx + gap * 45]/2 + src[mIdx + gap * 46]/2 - src[mIdx + gap * 48]/2 - src[mIdx + gap * 49]/2 - src[mIdx + gap * 50]/2 - src[mIdx + gap * 51]/2 - src[mIdx + gap * 52]/2 - src[mIdx + gap * 53]/2 - src[mIdx + gap * 54]/2 + src[mIdx + gap * 8] + src[mIdx + gap * 9];
dst[rIdx + outW * 1 + 1]  =  -src[mIdx + gap * 10] + 2*src[mIdx + gap * 11] - 2*src[mIdx + gap * 12] + src[mIdx + gap * 13]/2 - src[mIdx + gap * 14]/2 - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 2*src[mIdx + gap * 19] + 2*src[mIdx + gap * 20] - src[mIdx + gap * 21]/2 + src[mIdx + gap * 22]/2 + 2*src[mIdx + gap * 25] - 2*src[mIdx + gap * 26] + 4*src[mIdx + gap * 27] - 4*src[mIdx + gap * 28] + src[mIdx + gap * 29] - src[mIdx + gap * 30] - 2*src[mIdx + gap * 33] + 2*src[mIdx + gap * 34] - 4*src[mIdx + gap * 35] + 4*src[mIdx + gap * 36] - src[mIdx + gap * 37] + src[mIdx + gap * 38] + src[mIdx + gap * 41]/2 - src[mIdx + gap * 42]/2 + src[mIdx + gap * 43] - src[mIdx + gap * 44] + src[mIdx + gap * 45]/4 - src[mIdx + gap * 46]/4 - src[mIdx + gap * 49]/2 + src[mIdx + gap * 50]/2 - src[mIdx + gap * 51] + src[mIdx + gap * 52] - src[mIdx + gap * 53]/4 + src[mIdx + gap * 54]/4 + src[mIdx + gap * 9];
dst[rIdx + outW * 1 + 2]  =  src[mIdx + gap * 10] + 4*src[mIdx + gap * 11] + 4*src[mIdx + gap * 12] + src[mIdx + gap * 13]/4 + src[mIdx + gap * 14]/4 - src[mIdx + gap * 17] - src[mIdx + gap * 18] - 4*src[mIdx + gap * 19] - 4*src[mIdx + gap * 20] - src[mIdx + gap * 21]/4 - src[mIdx + gap * 22]/4 + 2*src[mIdx + gap * 25] + 2*src[mIdx + gap * 26] + 8*src[mIdx + gap * 27] + 8*src[mIdx + gap * 28] + src[mIdx + gap * 29]/2 + src[mIdx + gap * 30]/2 - 2*src[mIdx + gap * 33] - 2*src[mIdx + gap * 34] - 8*src[mIdx + gap * 35] - 8*src[mIdx + gap * 36] - src[mIdx + gap * 37]/2 - src[mIdx + gap * 38]/2 + src[mIdx + gap * 41]/2 + src[mIdx + gap * 42]/2 + 2*src[mIdx + gap * 43] + 2*src[mIdx + gap * 44] + src[mIdx + gap * 45]/8 + src[mIdx + gap * 46]/8 - src[mIdx + gap * 49]/2 - src[mIdx + gap * 50]/2 - 2*src[mIdx + gap * 51] - 2*src[mIdx + gap * 52] - src[mIdx + gap * 53]/8 - src[mIdx + gap * 54]/8 + src[mIdx + gap * 9];
dst[rIdx + outW * 1 + 3]  =  -src[mIdx + gap * 10] + 8*src[mIdx + gap * 11] - 8*src[mIdx + gap * 12] + src[mIdx + gap * 13]/8 - src[mIdx + gap * 14]/8 - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 8*src[mIdx + gap * 19] + 8*src[mIdx + gap * 20] - src[mIdx + gap * 21]/8 + src[mIdx + gap * 22]/8 + 2*src[mIdx + gap * 25] - 2*src[mIdx + gap * 26] + 16*src[mIdx + gap * 27] - 16*src[mIdx + gap * 28] + src[mIdx + gap * 29]/4 - src[mIdx + gap * 30]/4 - 2*src[mIdx + gap * 33] + 2*src[mIdx + gap * 34] - 16*src[mIdx + gap * 35] + 16*src[mIdx + gap * 36] - src[mIdx + gap * 37]/4 + src[mIdx + gap * 38]/4 + src[mIdx + gap * 41]/2 - src[mIdx + gap * 42]/2 + 4*src[mIdx + gap * 43] - 4*src[mIdx + gap * 44] + src[mIdx + gap * 45]/16 - src[mIdx + gap * 46]/16 - src[mIdx + gap * 49]/2 + src[mIdx + gap * 50]/2 - 4*src[mIdx + gap * 51] + 4*src[mIdx + gap * 52] - src[mIdx + gap * 53]/16 + src[mIdx + gap * 54]/16 + src[mIdx + gap * 9];
dst[rIdx + outW * 1 + 4]  =  src[mIdx + gap * 10] + 16*src[mIdx + gap * 11] + 16*src[mIdx + gap * 12] + src[mIdx + gap * 13]/16 + src[mIdx + gap * 14]/16 - src[mIdx + gap * 17] - src[mIdx + gap * 18] - 16*src[mIdx + gap * 19] - 16*src[mIdx + gap * 20] - src[mIdx + gap * 21]/16 - src[mIdx + gap * 22]/16 + 2*src[mIdx + gap * 25] + 2*src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] + 32*src[mIdx + gap * 28] + src[mIdx + gap * 29]/8 + src[mIdx + gap * 30]/8 - 2*src[mIdx + gap * 33] - 2*src[mIdx + gap * 34] - 32*src[mIdx + gap * 35] - 32*src[mIdx + gap * 36] - src[mIdx + gap * 37]/8 - src[mIdx + gap * 38]/8 + src[mIdx + gap * 41]/2 + src[mIdx + gap * 42]/2 + 8*src[mIdx + gap * 43] + 8*src[mIdx + gap * 44] + src[mIdx + gap * 45]/32 + src[mIdx + gap * 46]/32 - src[mIdx + gap * 49]/2 - src[mIdx + gap * 50]/2 - 8*src[mIdx + gap * 51] - 8*src[mIdx + gap * 52] - src[mIdx + gap * 53]/32 - src[mIdx + gap * 54]/32 + src[mIdx + gap * 9];
dst[rIdx + outW * 1 + 5]  =  -src[mIdx + gap * 10] + 32*src[mIdx + gap * 11] - 32*src[mIdx + gap * 12] + src[mIdx + gap * 13]/32 - src[mIdx + gap * 14]/32 + src[mIdx + gap * 15] - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 32*src[mIdx + gap * 19] + 32*src[mIdx + gap * 20] - src[mIdx + gap * 21]/32 + src[mIdx + gap * 22]/32 - src[mIdx + gap * 23] + 2*src[mIdx + gap * 25] - 2*src[mIdx + gap * 26] + 64*src[mIdx + gap * 27] - 64*src[mIdx + gap * 28] + src[mIdx + gap * 29]/16 - src[mIdx + gap * 30]/16 + 2*src[mIdx + gap * 31] - 2*src[mIdx + gap * 33] + 2*src[mIdx + gap * 34] - 64*src[mIdx + gap * 35] + 64*src[mIdx + gap * 36] - src[mIdx + gap * 37]/16 + src[mIdx + gap * 38]/16 - 2*src[mIdx + gap * 39] + src[mIdx + gap * 41]/2 - src[mIdx + gap * 42]/2 + 16*src[mIdx + gap * 43] - 16*src[mIdx + gap * 44] + src[mIdx + gap * 45]/64 - src[mIdx + gap * 46]/64 + src[mIdx + gap * 47]/2 - src[mIdx + gap * 49]/2 + src[mIdx + gap * 50]/2 - 16*src[mIdx + gap * 51] + 16*src[mIdx + gap * 52] - src[mIdx + gap * 53]/64 + src[mIdx + gap * 54]/64 - src[mIdx + gap * 55]/2 + src[mIdx + gap * 9];

dst[rIdx + outW * 2 + 0]  =  src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + src[mIdx + gap * 16] + src[mIdx + gap * 17] + src[mIdx + gap * 18] + src[mIdx + gap * 19] + src[mIdx + gap * 20] + src[mIdx + gap * 21] + src[mIdx + gap * 22] + 4*src[mIdx + gap * 24] + 4*src[mIdx + gap * 25] + 4*src[mIdx + gap * 26] + 4*src[mIdx + gap * 27] + 4*src[mIdx + gap * 28] + 4*src[mIdx + gap * 29] + 4*src[mIdx + gap * 30] + 4*src[mIdx + gap * 32] + 4*src[mIdx + gap * 33] + 4*src[mIdx + gap * 34] + 4*src[mIdx + gap * 35] + 4*src[mIdx + gap * 36] + 4*src[mIdx + gap * 37] + 4*src[mIdx + gap * 38] + src[mIdx + gap * 40]/4 + src[mIdx + gap * 41]/4 + src[mIdx + gap * 42]/4 + src[mIdx + gap * 43]/4 + src[mIdx + gap * 44]/4 + src[mIdx + gap * 45]/4 + src[mIdx + gap * 46]/4 + src[mIdx + gap * 48]/4 + src[mIdx + gap * 49]/4 + src[mIdx + gap * 50]/4 + src[mIdx + gap * 51]/4 + src[mIdx + gap * 52]/4 + src[mIdx + gap * 53]/4 + src[mIdx + gap * 54]/4 + src[mIdx + gap * 8] + src[mIdx + gap * 9];
dst[rIdx + outW * 2 + 1]  =  -src[mIdx + gap * 10] + 2*src[mIdx + gap * 11] - 2*src[mIdx + gap * 12] + src[mIdx + gap * 13]/2 - src[mIdx + gap * 14]/2 + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 2*src[mIdx + gap * 19] - 2*src[mIdx + gap * 20] + src[mIdx + gap * 21]/2 - src[mIdx + gap * 22]/2 + 4*src[mIdx + gap * 25] - 4*src[mIdx + gap * 26] + 8*src[mIdx + gap * 27] - 8*src[mIdx + gap * 28] + 2*src[mIdx + gap * 29] - 2*src[mIdx + gap * 30] + 4*src[mIdx + gap * 33] - 4*src[mIdx + gap * 34] + 8*src[mIdx + gap * 35] - 8*src[mIdx + gap * 36] + 2*src[mIdx + gap * 37] - 2*src[mIdx + gap * 38] + src[mIdx + gap * 41]/4 - src[mIdx + gap * 42]/4 + src[mIdx + gap * 43]/2 - src[mIdx + gap * 44]/2 + src[mIdx + gap * 45]/8 - src[mIdx + gap * 46]/8 + src[mIdx + gap * 49]/4 - src[mIdx + gap * 50]/4 + src[mIdx + gap * 51]/2 - src[mIdx + gap * 52]/2 + src[mIdx + gap * 53]/8 - src[mIdx + gap * 54]/8 + src[mIdx + gap * 9];
dst[rIdx + outW * 2 + 2]  =  src[mIdx + gap * 10] + 4*src[mIdx + gap * 11] + 4*src[mIdx + gap * 12] + src[mIdx + gap * 13]/4 + src[mIdx + gap * 14]/4 + src[mIdx + gap * 17] + src[mIdx + gap * 18] + 4*src[mIdx + gap * 19] + 4*src[mIdx + gap * 20] + src[mIdx + gap * 21]/4 + src[mIdx + gap * 22]/4 + 4*src[mIdx + gap * 25] + 4*src[mIdx + gap * 26] + 16*src[mIdx + gap * 27] + 16*src[mIdx + gap * 28] + src[mIdx + gap * 29] + src[mIdx + gap * 30] + 4*src[mIdx + gap * 33] + 4*src[mIdx + gap * 34] + 16*src[mIdx + gap * 35] + 16*src[mIdx + gap * 36] + src[mIdx + gap * 37] + src[mIdx + gap * 38] + src[mIdx + gap * 41]/4 + src[mIdx + gap * 42]/4 + src[mIdx + gap * 43] + src[mIdx + gap * 44] + src[mIdx + gap * 45]/16 + src[mIdx + gap * 46]/16 + src[mIdx + gap * 49]/4 + src[mIdx + gap * 50]/4 + src[mIdx + gap * 51] + src[mIdx + gap * 52] + src[mIdx + gap * 53]/16 + src[mIdx + gap * 54]/16 + src[mIdx + gap * 9];
dst[rIdx + outW * 2 + 3]  =  -src[mIdx + gap * 10] + 8*src[mIdx + gap * 11] - 8*src[mIdx + gap * 12] + src[mIdx + gap * 13]/8 - src[mIdx + gap * 14]/8 + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 8*src[mIdx + gap * 19] - 8*src[mIdx + gap * 20] + src[mIdx + gap * 21]/8 - src[mIdx + gap * 22]/8 + 4*src[mIdx + gap * 25] - 4*src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] - 32*src[mIdx + gap * 28] + src[mIdx + gap * 29]/2 - src[mIdx + gap * 30]/2 + 4*src[mIdx + gap * 33] - 4*src[mIdx + gap * 34] + 32*src[mIdx + gap * 35] - 32*src[mIdx + gap * 36] + src[mIdx + gap * 37]/2 - src[mIdx + gap * 38]/2 + src[mIdx + gap * 41]/4 - src[mIdx + gap * 42]/4 + 2*src[mIdx + gap * 43] - 2*src[mIdx + gap * 44] + src[mIdx + gap * 45]/32 - src[mIdx + gap * 46]/32 + src[mIdx + gap * 49]/4 - src[mIdx + gap * 50]/4 + 2*src[mIdx + gap * 51] - 2*src[mIdx + gap * 52] + src[mIdx + gap * 53]/32 - src[mIdx + gap * 54]/32 + src[mIdx + gap * 9];
dst[rIdx + outW * 2 + 4]  =  src[mIdx + gap * 10] + 16*src[mIdx + gap * 11] + 16*src[mIdx + gap * 12] + src[mIdx + gap * 13]/16 + src[mIdx + gap * 14]/16 + src[mIdx + gap * 17] + src[mIdx + gap * 18] + 16*src[mIdx + gap * 19] + 16*src[mIdx + gap * 20] + src[mIdx + gap * 21]/16 + src[mIdx + gap * 22]/16 + 4*src[mIdx + gap * 25] + 4*src[mIdx + gap * 26] + 64*src[mIdx + gap * 27] + 64*src[mIdx + gap * 28] + src[mIdx + gap * 29]/4 + src[mIdx + gap * 30]/4 + 4*src[mIdx + gap * 33] + 4*src[mIdx + gap * 34] + 64*src[mIdx + gap * 35] + 64*src[mIdx + gap * 36] + src[mIdx + gap * 37]/4 + src[mIdx + gap * 38]/4 + src[mIdx + gap * 41]/4 + src[mIdx + gap * 42]/4 + 4*src[mIdx + gap * 43] + 4*src[mIdx + gap * 44] + src[mIdx + gap * 45]/64 + src[mIdx + gap * 46]/64 + src[mIdx + gap * 49]/4 + src[mIdx + gap * 50]/4 + 4*src[mIdx + gap * 51] + 4*src[mIdx + gap * 52] + src[mIdx + gap * 53]/64 + src[mIdx + gap * 54]/64 + src[mIdx + gap * 9];
dst[rIdx + outW * 2 + 5]  =  -src[mIdx + gap * 10] + 32*src[mIdx + gap * 11] - 32*src[mIdx + gap * 12] + src[mIdx + gap * 13]/32 - src[mIdx + gap * 14]/32 + src[mIdx + gap * 15] + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 32*src[mIdx + gap * 19] - 32*src[mIdx + gap * 20] + src[mIdx + gap * 21]/32 - src[mIdx + gap * 22]/32 + src[mIdx + gap * 23] + 4*src[mIdx + gap * 25] - 4*src[mIdx + gap * 26] + 128*src[mIdx + gap * 27] - 128*src[mIdx + gap * 28] + src[mIdx + gap * 29]/8 - src[mIdx + gap * 30]/8 + 4*src[mIdx + gap * 31] + 4*src[mIdx + gap * 33] - 4*src[mIdx + gap * 34] + 128*src[mIdx + gap * 35] - 128*src[mIdx + gap * 36] + src[mIdx + gap * 37]/8 - src[mIdx + gap * 38]/8 + 4*src[mIdx + gap * 39] + src[mIdx + gap * 41]/4 - src[mIdx + gap * 42]/4 + 8*src[mIdx + gap * 43] - 8*src[mIdx + gap * 44] + src[mIdx + gap * 45]/128 - src[mIdx + gap * 46]/128 + src[mIdx + gap * 47]/4 + src[mIdx + gap * 49]/4 - src[mIdx + gap * 50]/4 + 8*src[mIdx + gap * 51] - 8*src[mIdx + gap * 52] + src[mIdx + gap * 53]/128 - src[mIdx + gap * 54]/128 + src[mIdx + gap * 55]/4 + src[mIdx + gap * 9];

dst[rIdx + outW * 3 + 0]  =  src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] - src[mIdx + gap * 16] - src[mIdx + gap * 17] - src[mIdx + gap * 18] - src[mIdx + gap * 19] - src[mIdx + gap * 20] - src[mIdx + gap * 21] - src[mIdx + gap * 22] + 8*src[mIdx + gap * 24] + 8*src[mIdx + gap * 25] + 8*src[mIdx + gap * 26] + 8*src[mIdx + gap * 27] + 8*src[mIdx + gap * 28] + 8*src[mIdx + gap * 29] + 8*src[mIdx + gap * 30] - 8*src[mIdx + gap * 32] - 8*src[mIdx + gap * 33] - 8*src[mIdx + gap * 34] - 8*src[mIdx + gap * 35] - 8*src[mIdx + gap * 36] - 8*src[mIdx + gap * 37] - 8*src[mIdx + gap * 38] + src[mIdx + gap * 40]/8 + src[mIdx + gap * 41]/8 + src[mIdx + gap * 42]/8 + src[mIdx + gap * 43]/8 + src[mIdx + gap * 44]/8 + src[mIdx + gap * 45]/8 + src[mIdx + gap * 46]/8 - src[mIdx + gap * 48]/8 - src[mIdx + gap * 49]/8 - src[mIdx + gap * 50]/8 - src[mIdx + gap * 51]/8 - src[mIdx + gap * 52]/8 - src[mIdx + gap * 53]/8 - src[mIdx + gap * 54]/8 + src[mIdx + gap * 8] + src[mIdx + gap * 9];
dst[rIdx + outW * 3 + 1]  =  -src[mIdx + gap * 10] + 2*src[mIdx + gap * 11] - 2*src[mIdx + gap * 12] + src[mIdx + gap * 13]/2 - src[mIdx + gap * 14]/2 - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 2*src[mIdx + gap * 19] + 2*src[mIdx + gap * 20] - src[mIdx + gap * 21]/2 + src[mIdx + gap * 22]/2 + 8*src[mIdx + gap * 25] - 8*src[mIdx + gap * 26] + 16*src[mIdx + gap * 27] - 16*src[mIdx + gap * 28] + 4*src[mIdx + gap * 29] - 4*src[mIdx + gap * 30] - 8*src[mIdx + gap * 33] + 8*src[mIdx + gap * 34] - 16*src[mIdx + gap * 35] + 16*src[mIdx + gap * 36] - 4*src[mIdx + gap * 37] + 4*src[mIdx + gap * 38] + src[mIdx + gap * 41]/8 - src[mIdx + gap * 42]/8 + src[mIdx + gap * 43]/4 - src[mIdx + gap * 44]/4 + src[mIdx + gap * 45]/16 - src[mIdx + gap * 46]/16 - src[mIdx + gap * 49]/8 + src[mIdx + gap * 50]/8 - src[mIdx + gap * 51]/4 + src[mIdx + gap * 52]/4 - src[mIdx + gap * 53]/16 + src[mIdx + gap * 54]/16 + src[mIdx + gap * 9];
dst[rIdx + outW * 3 + 2]  =  src[mIdx + gap * 10] + 4*src[mIdx + gap * 11] + 4*src[mIdx + gap * 12] + src[mIdx + gap * 13]/4 + src[mIdx + gap * 14]/4 - src[mIdx + gap * 17] - src[mIdx + gap * 18] - 4*src[mIdx + gap * 19] - 4*src[mIdx + gap * 20] - src[mIdx + gap * 21]/4 - src[mIdx + gap * 22]/4 + 8*src[mIdx + gap * 25] + 8*src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] + 32*src[mIdx + gap * 28] + 2*src[mIdx + gap * 29] + 2*src[mIdx + gap * 30] - 8*src[mIdx + gap * 33] - 8*src[mIdx + gap * 34] - 32*src[mIdx + gap * 35] - 32*src[mIdx + gap * 36] - 2*src[mIdx + gap * 37] - 2*src[mIdx + gap * 38] + src[mIdx + gap * 41]/8 + src[mIdx + gap * 42]/8 + src[mIdx + gap * 43]/2 + src[mIdx + gap * 44]/2 + src[mIdx + gap * 45]/32 + src[mIdx + gap * 46]/32 - src[mIdx + gap * 49]/8 - src[mIdx + gap * 50]/8 - src[mIdx + gap * 51]/2 - src[mIdx + gap * 52]/2 - src[mIdx + gap * 53]/32 - src[mIdx + gap * 54]/32 + src[mIdx + gap * 9];
dst[rIdx + outW * 3 + 3]  =  -src[mIdx + gap * 10] + 8*src[mIdx + gap * 11] - 8*src[mIdx + gap * 12] + src[mIdx + gap * 13]/8 - src[mIdx + gap * 14]/8 - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 8*src[mIdx + gap * 19] + 8*src[mIdx + gap * 20] - src[mIdx + gap * 21]/8 + src[mIdx + gap * 22]/8 + 8*src[mIdx + gap * 25] - 8*src[mIdx + gap * 26] + 64*src[mIdx + gap * 27] - 64*src[mIdx + gap * 28] + src[mIdx + gap * 29] - src[mIdx + gap * 30] - 8*src[mIdx + gap * 33] + 8*src[mIdx + gap * 34] - 64*src[mIdx + gap * 35] + 64*src[mIdx + gap * 36] - src[mIdx + gap * 37] + src[mIdx + gap * 38] + src[mIdx + gap * 41]/8 - src[mIdx + gap * 42]/8 + src[mIdx + gap * 43] - src[mIdx + gap * 44] + src[mIdx + gap * 45]/64 - src[mIdx + gap * 46]/64 - src[mIdx + gap * 49]/8 + src[mIdx + gap * 50]/8 - src[mIdx + gap * 51] + src[mIdx + gap * 52] - src[mIdx + gap * 53]/64 + src[mIdx + gap * 54]/64 + src[mIdx + gap * 9];
dst[rIdx + outW * 3 + 4]  =  src[mIdx + gap * 10] + 16*src[mIdx + gap * 11] + 16*src[mIdx + gap * 12] + src[mIdx + gap * 13]/16 + src[mIdx + gap * 14]/16 - src[mIdx + gap * 17] - src[mIdx + gap * 18] - 16*src[mIdx + gap * 19] - 16*src[mIdx + gap * 20] - src[mIdx + gap * 21]/16 - src[mIdx + gap * 22]/16 + 8*src[mIdx + gap * 25] + 8*src[mIdx + gap * 26] + 128*src[mIdx + gap * 27] + 128*src[mIdx + gap * 28] + src[mIdx + gap * 29]/2 + src[mIdx + gap * 30]/2 - 8*src[mIdx + gap * 33] - 8*src[mIdx + gap * 34] - 128*src[mIdx + gap * 35] - 128*src[mIdx + gap * 36] - src[mIdx + gap * 37]/2 - src[mIdx + gap * 38]/2 + src[mIdx + gap * 41]/8 + src[mIdx + gap * 42]/8 + 2*src[mIdx + gap * 43] + 2*src[mIdx + gap * 44] + src[mIdx + gap * 45]/128 + src[mIdx + gap * 46]/128 - src[mIdx + gap * 49]/8 - src[mIdx + gap * 50]/8 - 2*src[mIdx + gap * 51] - 2*src[mIdx + gap * 52] - src[mIdx + gap * 53]/128 - src[mIdx + gap * 54]/128 + src[mIdx + gap * 9];
dst[rIdx + outW * 3 + 5]  =  -src[mIdx + gap * 10] + 32*src[mIdx + gap * 11] - 32*src[mIdx + gap * 12] + src[mIdx + gap * 13]/32 - src[mIdx + gap * 14]/32 + src[mIdx + gap * 15] - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 32*src[mIdx + gap * 19] + 32*src[mIdx + gap * 20] - src[mIdx + gap * 21]/32 + src[mIdx + gap * 22]/32 - src[mIdx + gap * 23] + 8*src[mIdx + gap * 25] - 8*src[mIdx + gap * 26] + 256*src[mIdx + gap * 27] - 256*src[mIdx + gap * 28] + src[mIdx + gap * 29]/4 - src[mIdx + gap * 30]/4 + 8*src[mIdx + gap * 31] - 8*src[mIdx + gap * 33] + 8*src[mIdx + gap * 34] - 256*src[mIdx + gap * 35] + 256*src[mIdx + gap * 36] - src[mIdx + gap * 37]/4 + src[mIdx + gap * 38]/4 - 8*src[mIdx + gap * 39] + src[mIdx + gap * 41]/8 - src[mIdx + gap * 42]/8 + 4*src[mIdx + gap * 43] - 4*src[mIdx + gap * 44] + src[mIdx + gap * 45]/256 - src[mIdx + gap * 46]/256 + src[mIdx + gap * 47]/8 - src[mIdx + gap * 49]/8 + src[mIdx + gap * 50]/8 - 4*src[mIdx + gap * 51] + 4*src[mIdx + gap * 52] - src[mIdx + gap * 53]/256 + src[mIdx + gap * 54]/256 - src[mIdx + gap * 55]/8 + src[mIdx + gap * 9];

dst[rIdx + outW * 4 + 0]  =  src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] + src[mIdx + gap * 16] + src[mIdx + gap * 17] + src[mIdx + gap * 18] + src[mIdx + gap * 19] + src[mIdx + gap * 20] + src[mIdx + gap * 21] + src[mIdx + gap * 22] + 16*src[mIdx + gap * 24] + 16*src[mIdx + gap * 25] + 16*src[mIdx + gap * 26] + 16*src[mIdx + gap * 27] + 16*src[mIdx + gap * 28] + 16*src[mIdx + gap * 29] + 16*src[mIdx + gap * 30] + 16*src[mIdx + gap * 32] + 16*src[mIdx + gap * 33] + 16*src[mIdx + gap * 34] + 16*src[mIdx + gap * 35] + 16*src[mIdx + gap * 36] + 16*src[mIdx + gap * 37] + 16*src[mIdx + gap * 38] + src[mIdx + gap * 40]/16 + src[mIdx + gap * 41]/16 + src[mIdx + gap * 42]/16 + src[mIdx + gap * 43]/16 + src[mIdx + gap * 44]/16 + src[mIdx + gap * 45]/16 + src[mIdx + gap * 46]/16 + src[mIdx + gap * 48]/16 + src[mIdx + gap * 49]/16 + src[mIdx + gap * 50]/16 + src[mIdx + gap * 51]/16 + src[mIdx + gap * 52]/16 + src[mIdx + gap * 53]/16 + src[mIdx + gap * 54]/16 + src[mIdx + gap * 8] + src[mIdx + gap * 9];
dst[rIdx + outW * 4 + 1]  =  -src[mIdx + gap * 10] + 2*src[mIdx + gap * 11] - 2*src[mIdx + gap * 12] + src[mIdx + gap * 13]/2 - src[mIdx + gap * 14]/2 + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 2*src[mIdx + gap * 19] - 2*src[mIdx + gap * 20] + src[mIdx + gap * 21]/2 - src[mIdx + gap * 22]/2 + 16*src[mIdx + gap * 25] - 16*src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] - 32*src[mIdx + gap * 28] + 8*src[mIdx + gap * 29] - 8*src[mIdx + gap * 30] + 16*src[mIdx + gap * 33] - 16*src[mIdx + gap * 34] + 32*src[mIdx + gap * 35] - 32*src[mIdx + gap * 36] + 8*src[mIdx + gap * 37] - 8*src[mIdx + gap * 38] + src[mIdx + gap * 41]/16 - src[mIdx + gap * 42]/16 + src[mIdx + gap * 43]/8 - src[mIdx + gap * 44]/8 + src[mIdx + gap * 45]/32 - src[mIdx + gap * 46]/32 + src[mIdx + gap * 49]/16 - src[mIdx + gap * 50]/16 + src[mIdx + gap * 51]/8 - src[mIdx + gap * 52]/8 + src[mIdx + gap * 53]/32 - src[mIdx + gap * 54]/32 + src[mIdx + gap * 9];
dst[rIdx + outW * 4 + 2]  =  src[mIdx + gap * 10] + 4*src[mIdx + gap * 11] + 4*src[mIdx + gap * 12] + src[mIdx + gap * 13]/4 + src[mIdx + gap * 14]/4 + src[mIdx + gap * 17] + src[mIdx + gap * 18] + 4*src[mIdx + gap * 19] + 4*src[mIdx + gap * 20] + src[mIdx + gap * 21]/4 + src[mIdx + gap * 22]/4 + 16*src[mIdx + gap * 25] + 16*src[mIdx + gap * 26] + 64*src[mIdx + gap * 27] + 64*src[mIdx + gap * 28] + 4*src[mIdx + gap * 29] + 4*src[mIdx + gap * 30] + 16*src[mIdx + gap * 33] + 16*src[mIdx + gap * 34] + 64*src[mIdx + gap * 35] + 64*src[mIdx + gap * 36] + 4*src[mIdx + gap * 37] + 4*src[mIdx + gap * 38] + src[mIdx + gap * 41]/16 + src[mIdx + gap * 42]/16 + src[mIdx + gap * 43]/4 + src[mIdx + gap * 44]/4 + src[mIdx + gap * 45]/64 + src[mIdx + gap * 46]/64 + src[mIdx + gap * 49]/16 + src[mIdx + gap * 50]/16 + src[mIdx + gap * 51]/4 + src[mIdx + gap * 52]/4 + src[mIdx + gap * 53]/64 + src[mIdx + gap * 54]/64 + src[mIdx + gap * 9];
dst[rIdx + outW * 4 + 3]  =  -src[mIdx + gap * 10] + 8*src[mIdx + gap * 11] - 8*src[mIdx + gap * 12] + src[mIdx + gap * 13]/8 - src[mIdx + gap * 14]/8 + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 8*src[mIdx + gap * 19] - 8*src[mIdx + gap * 20] + src[mIdx + gap * 21]/8 - src[mIdx + gap * 22]/8 + 16*src[mIdx + gap * 25] - 16*src[mIdx + gap * 26] + 128*src[mIdx + gap * 27] - 128*src[mIdx + gap * 28] + 2*src[mIdx + gap * 29] - 2*src[mIdx + gap * 30] + 16*src[mIdx + gap * 33] - 16*src[mIdx + gap * 34] + 128*src[mIdx + gap * 35] - 128*src[mIdx + gap * 36] + 2*src[mIdx + gap * 37] - 2*src[mIdx + gap * 38] + src[mIdx + gap * 41]/16 - src[mIdx + gap * 42]/16 + src[mIdx + gap * 43]/2 - src[mIdx + gap * 44]/2 + src[mIdx + gap * 45]/128 - src[mIdx + gap * 46]/128 + src[mIdx + gap * 49]/16 - src[mIdx + gap * 50]/16 + src[mIdx + gap * 51]/2 - src[mIdx + gap * 52]/2 + src[mIdx + gap * 53]/128 - src[mIdx + gap * 54]/128 + src[mIdx + gap * 9];
dst[rIdx + outW * 4 + 4]  =  src[mIdx + gap * 10] + 16*src[mIdx + gap * 11] + 16*src[mIdx + gap * 12] + src[mIdx + gap * 13]/16 + src[mIdx + gap * 14]/16 + src[mIdx + gap * 17] + src[mIdx + gap * 18] + 16*src[mIdx + gap * 19] + 16*src[mIdx + gap * 20] + src[mIdx + gap * 21]/16 + src[mIdx + gap * 22]/16 + 16*src[mIdx + gap * 25] + 16*src[mIdx + gap * 26] + 256*src[mIdx + gap * 27] + 256*src[mIdx + gap * 28] + src[mIdx + gap * 29] + src[mIdx + gap * 30] + 16*src[mIdx + gap * 33] + 16*src[mIdx + gap * 34] + 256*src[mIdx + gap * 35] + 256*src[mIdx + gap * 36] + src[mIdx + gap * 37] + src[mIdx + gap * 38] + src[mIdx + gap * 41]/16 + src[mIdx + gap * 42]/16 + src[mIdx + gap * 43] + src[mIdx + gap * 44] + src[mIdx + gap * 45]/256 + src[mIdx + gap * 46]/256 + src[mIdx + gap * 49]/16 + src[mIdx + gap * 50]/16 + src[mIdx + gap * 51] + src[mIdx + gap * 52] + src[mIdx + gap * 53]/256 + src[mIdx + gap * 54]/256 + src[mIdx + gap * 9];
dst[rIdx + outW * 4 + 5]  =  -src[mIdx + gap * 10] + 32*src[mIdx + gap * 11] - 32*src[mIdx + gap * 12] + src[mIdx + gap * 13]/32 - src[mIdx + gap * 14]/32 + src[mIdx + gap * 15] + src[mIdx + gap * 17] - src[mIdx + gap * 18] + 32*src[mIdx + gap * 19] - 32*src[mIdx + gap * 20] + src[mIdx + gap * 21]/32 - src[mIdx + gap * 22]/32 + src[mIdx + gap * 23] + 16*src[mIdx + gap * 25] - 16*src[mIdx + gap * 26] + 512*src[mIdx + gap * 27] - 512*src[mIdx + gap * 28] + src[mIdx + gap * 29]/2 - src[mIdx + gap * 30]/2 + 16*src[mIdx + gap * 31] + 16*src[mIdx + gap * 33] - 16*src[mIdx + gap * 34] + 512*src[mIdx + gap * 35] - 512*src[mIdx + gap * 36] + src[mIdx + gap * 37]/2 - src[mIdx + gap * 38]/2 + 16*src[mIdx + gap * 39] + src[mIdx + gap * 41]/16 - src[mIdx + gap * 42]/16 + 2*src[mIdx + gap * 43] - 2*src[mIdx + gap * 44] + src[mIdx + gap * 45]/512 - src[mIdx + gap * 46]/512 + src[mIdx + gap * 47]/16 + src[mIdx + gap * 49]/16 - src[mIdx + gap * 50]/16 + 2*src[mIdx + gap * 51] - 2*src[mIdx + gap * 52] + src[mIdx + gap * 53]/512 - src[mIdx + gap * 54]/512 + src[mIdx + gap * 55]/16 + src[mIdx + gap * 9];

dst[rIdx + outW * 5 + 0]  =  src[mIdx + gap * 10] + src[mIdx + gap * 11] + src[mIdx + gap * 12] + src[mIdx + gap * 13] + src[mIdx + gap * 14] - src[mIdx + gap * 16] - src[mIdx + gap * 17] - src[mIdx + gap * 18] - src[mIdx + gap * 19] - src[mIdx + gap * 20] - src[mIdx + gap * 21] - src[mIdx + gap * 22] + 32*src[mIdx + gap * 24] + 32*src[mIdx + gap * 25] + 32*src[mIdx + gap * 26] + 32*src[mIdx + gap * 27] + 32*src[mIdx + gap * 28] + 32*src[mIdx + gap * 29] + 32*src[mIdx + gap * 30] - 32*src[mIdx + gap * 32] - 32*src[mIdx + gap * 33] - 32*src[mIdx + gap * 34] - 32*src[mIdx + gap * 35] - 32*src[mIdx + gap * 36] - 32*src[mIdx + gap * 37] - 32*src[mIdx + gap * 38] + src[mIdx + gap * 40]/32 + src[mIdx + gap * 41]/32 + src[mIdx + gap * 42]/32 + src[mIdx + gap * 43]/32 + src[mIdx + gap * 44]/32 + src[mIdx + gap * 45]/32 + src[mIdx + gap * 46]/32 - src[mIdx + gap * 48]/32 - src[mIdx + gap * 49]/32 - src[mIdx + gap * 50]/32 - src[mIdx + gap * 51]/32 - src[mIdx + gap * 52]/32 - src[mIdx + gap * 53]/32 - src[mIdx + gap * 54]/32 + src[mIdx + gap * 56] + src[mIdx + gap * 57] + src[mIdx + gap * 58] + src[mIdx + gap * 59] + src[mIdx + gap * 60] + src[mIdx + gap * 61] + src[mIdx + gap * 62] + src[mIdx + gap * 8] + src[mIdx + gap * 9];
dst[rIdx + outW * 5 + 1]  =  -src[mIdx + gap * 10] + 2*src[mIdx + gap * 11] - 2*src[mIdx + gap * 12] + src[mIdx + gap * 13]/2 - src[mIdx + gap * 14]/2 - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 2*src[mIdx + gap * 19] + 2*src[mIdx + gap * 20] - src[mIdx + gap * 21]/2 + src[mIdx + gap * 22]/2 + 32*src[mIdx + gap * 25] - 32*src[mIdx + gap * 26] + 64*src[mIdx + gap * 27] - 64*src[mIdx + gap * 28] + 16*src[mIdx + gap * 29] - 16*src[mIdx + gap * 30] - 32*src[mIdx + gap * 33] + 32*src[mIdx + gap * 34] - 64*src[mIdx + gap * 35] + 64*src[mIdx + gap * 36] - 16*src[mIdx + gap * 37] + 16*src[mIdx + gap * 38] + src[mIdx + gap * 41]/32 - src[mIdx + gap * 42]/32 + src[mIdx + gap * 43]/16 - src[mIdx + gap * 44]/16 + src[mIdx + gap * 45]/64 - src[mIdx + gap * 46]/64 - src[mIdx + gap * 49]/32 + src[mIdx + gap * 50]/32 - src[mIdx + gap * 51]/16 + src[mIdx + gap * 52]/16 - src[mIdx + gap * 53]/64 + src[mIdx + gap * 54]/64 + src[mIdx + gap * 57] - src[mIdx + gap * 58] + 2*src[mIdx + gap * 59] - 2*src[mIdx + gap * 60] + src[mIdx + gap * 61]/2 - src[mIdx + gap * 62]/2 + src[mIdx + gap * 9];
dst[rIdx + outW * 5 + 2]  =  src[mIdx + gap * 10] + 4*src[mIdx + gap * 11] + 4*src[mIdx + gap * 12] + src[mIdx + gap * 13]/4 + src[mIdx + gap * 14]/4 - src[mIdx + gap * 17] - src[mIdx + gap * 18] - 4*src[mIdx + gap * 19] - 4*src[mIdx + gap * 20] - src[mIdx + gap * 21]/4 - src[mIdx + gap * 22]/4 + 32*src[mIdx + gap * 25] + 32*src[mIdx + gap * 26] + 128*src[mIdx + gap * 27] + 128*src[mIdx + gap * 28] + 8*src[mIdx + gap * 29] + 8*src[mIdx + gap * 30] - 32*src[mIdx + gap * 33] - 32*src[mIdx + gap * 34] - 128*src[mIdx + gap * 35] - 128*src[mIdx + gap * 36] - 8*src[mIdx + gap * 37] - 8*src[mIdx + gap * 38] + src[mIdx + gap * 41]/32 + src[mIdx + gap * 42]/32 + src[mIdx + gap * 43]/8 + src[mIdx + gap * 44]/8 + src[mIdx + gap * 45]/128 + src[mIdx + gap * 46]/128 - src[mIdx + gap * 49]/32 - src[mIdx + gap * 50]/32 - src[mIdx + gap * 51]/8 - src[mIdx + gap * 52]/8 - src[mIdx + gap * 53]/128 - src[mIdx + gap * 54]/128 + src[mIdx + gap * 57] + src[mIdx + gap * 58] + 4*src[mIdx + gap * 59] + 4*src[mIdx + gap * 60] + src[mIdx + gap * 61]/4 + src[mIdx + gap * 62]/4 + src[mIdx + gap * 9];
dst[rIdx + outW * 5 + 3]  =  -src[mIdx + gap * 10] + 8*src[mIdx + gap * 11] - 8*src[mIdx + gap * 12] + src[mIdx + gap * 13]/8 - src[mIdx + gap * 14]/8 - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 8*src[mIdx + gap * 19] + 8*src[mIdx + gap * 20] - src[mIdx + gap * 21]/8 + src[mIdx + gap * 22]/8 + 32*src[mIdx + gap * 25] - 32*src[mIdx + gap * 26] + 256*src[mIdx + gap * 27] - 256*src[mIdx + gap * 28] + 4*src[mIdx + gap * 29] - 4*src[mIdx + gap * 30] - 32*src[mIdx + gap * 33] + 32*src[mIdx + gap * 34] - 256*src[mIdx + gap * 35] + 256*src[mIdx + gap * 36] - 4*src[mIdx + gap * 37] + 4*src[mIdx + gap * 38] + src[mIdx + gap * 41]/32 - src[mIdx + gap * 42]/32 + src[mIdx + gap * 43]/4 - src[mIdx + gap * 44]/4 + src[mIdx + gap * 45]/256 - src[mIdx + gap * 46]/256 - src[mIdx + gap * 49]/32 + src[mIdx + gap * 50]/32 - src[mIdx + gap * 51]/4 + src[mIdx + gap * 52]/4 - src[mIdx + gap * 53]/256 + src[mIdx + gap * 54]/256 + src[mIdx + gap * 57] - src[mIdx + gap * 58] + 8*src[mIdx + gap * 59] - 8*src[mIdx + gap * 60] + src[mIdx + gap * 61]/8 - src[mIdx + gap * 62]/8 + src[mIdx + gap * 9];
dst[rIdx + outW * 5 + 4]  =  src[mIdx + gap * 10] + 16*src[mIdx + gap * 11] + 16*src[mIdx + gap * 12] + src[mIdx + gap * 13]/16 + src[mIdx + gap * 14]/16 - src[mIdx + gap * 17] - src[mIdx + gap * 18] - 16*src[mIdx + gap * 19] - 16*src[mIdx + gap * 20] - src[mIdx + gap * 21]/16 - src[mIdx + gap * 22]/16 + 32*src[mIdx + gap * 25] + 32*src[mIdx + gap * 26] + 512*src[mIdx + gap * 27] + 512*src[mIdx + gap * 28] + 2*src[mIdx + gap * 29] + 2*src[mIdx + gap * 30] - 32*src[mIdx + gap * 33] - 32*src[mIdx + gap * 34] - 512*src[mIdx + gap * 35] - 512*src[mIdx + gap * 36] - 2*src[mIdx + gap * 37] - 2*src[mIdx + gap * 38] + src[mIdx + gap * 41]/32 + src[mIdx + gap * 42]/32 + src[mIdx + gap * 43]/2 + src[mIdx + gap * 44]/2 + src[mIdx + gap * 45]/512 + src[mIdx + gap * 46]/512 - src[mIdx + gap * 49]/32 - src[mIdx + gap * 50]/32 - src[mIdx + gap * 51]/2 - src[mIdx + gap * 52]/2 - src[mIdx + gap * 53]/512 - src[mIdx + gap * 54]/512 + src[mIdx + gap * 57] + src[mIdx + gap * 58] + 16*src[mIdx + gap * 59] + 16*src[mIdx + gap * 60] + src[mIdx + gap * 61]/16 + src[mIdx + gap * 62]/16 + src[mIdx + gap * 9];
dst[rIdx + outW * 5 + 5]  =  -src[mIdx + gap * 10] + 32*src[mIdx + gap * 11] - 32*src[mIdx + gap * 12] + src[mIdx + gap * 13]/32 - src[mIdx + gap * 14]/32 + src[mIdx + gap * 15] - src[mIdx + gap * 17] + src[mIdx + gap * 18] - 32*src[mIdx + gap * 19] + 32*src[mIdx + gap * 20] - src[mIdx + gap * 21]/32 + src[mIdx + gap * 22]/32 - src[mIdx + gap * 23] + 32*src[mIdx + gap * 25] - 32*src[mIdx + gap * 26] + 1024*src[mIdx + gap * 27] - 1024*src[mIdx + gap * 28] + src[mIdx + gap * 29] - src[mIdx + gap * 30] + 32*src[mIdx + gap * 31] - 32*src[mIdx + gap * 33] + 32*src[mIdx + gap * 34] - 1024*src[mIdx + gap * 35] + 1024*src[mIdx + gap * 36] - src[mIdx + gap * 37] + src[mIdx + gap * 38] - 32*src[mIdx + gap * 39] + src[mIdx + gap * 41]/32 - src[mIdx + gap * 42]/32 + src[mIdx + gap * 43] - src[mIdx + gap * 44] + src[mIdx + gap * 45]/1024 - src[mIdx + gap * 46]/1024 + src[mIdx + gap * 47]/32 - src[mIdx + gap * 49]/32 + src[mIdx + gap * 50]/32 - src[mIdx + gap * 51] + src[mIdx + gap * 52] - src[mIdx + gap * 53]/1024 + src[mIdx + gap * 54]/1024 - src[mIdx + gap * 55]/32 + src[mIdx + gap * 57] - src[mIdx + gap * 58] + 32*src[mIdx + gap * 59] - 32*src[mIdx + gap * 60] + src[mIdx + gap * 61]/32 - src[mIdx + gap * 62]/32 + src[mIdx + gap * 63] + src[mIdx + gap * 9];


	}
}



template <typename Dtype> 
__global__ void winoDstAddOpt_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{


	CUDA_KERNEL_LOOP(idx, tNums) {
		
		int highIdx = idx / (tileW * tileH);

		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;

		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

        float tmp; 
		float A[16]; 

		//// -- project ---- ///

	}


}


template <typename Dtype> 
__global__ void wino4x4DstAddOpt_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{


	CUDA_KERNEL_LOOP(idx, tNums) {
		
		int highIdx = idx / (tileW * tileH);

		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;

		int rIdx = highIdx * outW * outH + yIdx * outW * 4 + xIdx * 4;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;


		//// -- project ---- ///

float s[20], a[8];


a[0] = DSRC(7);
a[1] = DSRC(8);
a[2] = DSRC(9);
a[3] = DSRC(10);
a[4] = DSRC(13);
a[5] = DSRC(14);
a[6] = DSRC(15);
a[7] = DSRC(16);

s[0]  = a[0] + a[4];
s[8]  = a[0] - a[4];
s[1]  = a[1] + a[5];
s[9]  = a[1] - a[5];
s[2]  = a[2] + a[6];
s[10] = a[2] - a[6];
s[3]  = a[3] + a[7];
s[11] = a[3] - a[7];

a[0] = DSRC(19);
a[1] = DSRC(20);
a[2] = DSRC(21);
a[3] = DSRC(22);
a[4] = DSRC(25);
a[5] = DSRC(26);
a[6] = DSRC(27);
a[7] = DSRC(28);

s[4]  = a[0] + a[4];
s[12] = a[0] - a[4];
s[5]  = a[1] + a[5];
s[13] = a[1] - a[5];
s[6]  = a[2] + a[6];
s[14] = a[2] - a[6];
s[7]  = a[3] + a[7];
s[15] = a[3] - a[7];

a[0] = DSRC(1) + s[0] + s[4];
a[1] = DSRC(2) + s[1] + s[5];
a[2] = DSRC(3) + s[2] + s[6];
a[3] = DSRC(4) + s[3] + s[7];

dst[rIdx + outW * 0 + 0]  =  DSRC(0) + DSRC(6) + DSRC(12) + DSRC(18) + DSRC(24) + a[0] + a[1] + a[2] + a[3];
dst[rIdx + outW * 0 + 1]  =  a[0] - a[1] + 2*(a[2] - a[3]);
dst[rIdx + outW * 0 + 2]  =  a[0] + a[1] + 4*(a[2] + a[3]);

s[16] = DSRC(11);
s[17] = DSRC(17);
s[18] = DSRC(23);
s[19] = DSRC(29);
dst[rIdx + outW * 0 + 3]  =  DSRC(5) + s[16] + s[17] + s[18] + s[19] + a[0] - a[1] + 8*(a[2] - a[3]);

a[0] =  s[8] + 2*(s[12]);
a[1] =  s[9] + 2*(s[13]);
a[2] = s[10] + 2*(s[14]);
a[3] = s[11] + 2*(s[15]);

a[4] = s[0] + 4*(s[4]);
a[5] = s[1] + 4*(s[5]);
a[6] = s[2] + 4*(s[6]);
a[7] = s[3] + 4*(s[7]);

dst[rIdx + outW * 1 + 3]  = s[16] - s[17] + 2*(s[18] - s[19]) + a[0] - a[1] + 8*(a[2] - a[3]);
dst[rIdx + outW * 2 + 3]  = s[16] + s[17] + 4*(s[18] + s[19]) + a[4] - a[5] + 8*(a[6] - a[7]);

dst[rIdx + outW * 1 + 1]  = a[0] - a[1] + 2*(a[2] - a[3]);
dst[rIdx + outW * 1 + 2]  = a[0] + a[1] + 4*(a[2] + a[3]);

dst[rIdx + outW * 2 + 1]  = a[4] - a[5] + 2*(a[6] - a[7]);
dst[rIdx + outW * 2 + 2]  = a[4] + a[5] + 4*(a[6] + a[7]);

s[16] = DSRC(6);
s[17] = DSRC(12);
s[18] = DSRC(18);
s[19] = DSRC(24);

dst[rIdx + outW * 1 + 0]  =  s[16] - s[17] + 2*(s[18] - s[19]) + a[0] + a[1] + a[2] + a[3];
dst[rIdx + outW * 2 + 0]  =  s[16] + s[17] + 4*(s[18] + s[19]) + a[4] + a[5] + a[6] + a[7];

a[0] = s[8] + 8*s[12] + DSRC(31);
a[1] = s[9] + 8*s[13] + DSRC(32);
a[2] = s[10] + 8*s[14] + DSRC(33);
a[3] = s[11] + 8*s[15] + DSRC(34);

dst[rIdx + outW * 3 + 0]  =  s[16] - s[17] + 8*(s[18] - s[19]) + DSRC(30) + a[0] + a[1] + a[2] + a[3];
dst[rIdx + outW * 3 + 1]  =  a[0] - a[1] + 2*(a[2] - a[3]);
dst[rIdx + outW * 3 + 2]  =  a[0] + a[1] + 4*(a[2] + a[3]);
dst[rIdx + outW * 3 + 3]  =  DSRC(11) - DSRC(17) + 8*DSRC(23) - 8*DSRC(29) + DSRC(35) + a[0] - a[1] + 8*(a[2] - a[3]);

	}
}

template <typename Dtype> 
__global__ void wino6x6DstAddOpt_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{


	CUDA_KERNEL_LOOP(idx, tNums) {
		
		int highIdx = idx / (tileW * tileH);

		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;

		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;


		//// -- project ---- ///

	}
}


template <typename Dtype>
void winoWeight_gpu(const int num_inputs, const int num_outputs, 
					const Dtype *weight, Dtype *wino_weight, const int wino_tile_size )
{
	int num_kernels = num_inputs * num_outputs;

	if((wino_tile_size == 2) || (wino_tile_size == 12))
		winoWeight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 
	else if((wino_tile_size == 4) || (wino_tile_size == 14))
		wino4x4Weight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 
	else if((wino_tile_size == 6) || (wino_tile_size == 16))
		wino6x6Weight_gpu_kernel<Dtype><<<(num_kernels + WINO6_TH-1) / WINO6_TH,
			                         WINO6_TH>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 

}

template void winoWeight_gpu<float>(const int num_inputs, const int num_outputs, 
									const float *weight, float *wino_weight, const int wino_tile_size); 
template void winoWeight_gpu<double>(const int num_inputs, const int num_outputs, 
									const double *weight, double *wino_weight, const int wino_tile_size); 




template <typename Dtype>
void padSrc_gpu(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const Dtype *input, Dtype *input_pad, const int c_height, const int c_width)
{

	int num_kernels = batchs * num_inputs * c_height * c_width; 
	
	padSrc_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(input, input_pad, height, width, c_height, c_width, num_inputs, batchs, height_pad, 0, num_kernels); 

}

template void padSrc_gpu<float>(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const float *input, float *input_pad, const int c_height, const int c_width); 
template void padSrc_gpu<double>(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const double *input, double *input_pad, const int c_height, const int c_width); 


template <typename Dtype>
void winoSrc_gpu(const int batchs, const int num_inputs, const int tileH, const int tileW, 
				const int height, const int width, // include padding 
				const Dtype *m_matrix, Dtype *v_matrix, const int wino_tile_size)
{
	int num_kernels = batchs * num_inputs * tileH * tileW;

	if(wino_tile_size == 2)
	{
		winoSrc_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
				                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 12)
	{
		winoSrcAddOpt_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if (wino_tile_size == 4)
	{
		wino4x4Src_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
				                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 14)
	{
		winoSrcAddOpt_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 6)
	{
		wino6x6Src_gpu_kernel<Dtype><<<(num_kernels + WINO6_TH-1) / WINO6_TH,
				                         WINO6_TH>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 16)
	{
		wino6x6SrcAddOpt_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
}

template void winoSrc_gpu<float>(const int batchs, const int num_inputs, const int tileH, const int tileW, 
						const int height, const int width, // include padding 
						const float *m_matrix, float *v_matrix, const int wino_tile_size); 
template void winoSrc_gpu<double>(const int batchs, const int num_inputs, const int tileH, const int tileW, 
						const int height, const int width, // include padding 
						const double *m_matrix, double *v_matrix, const int wino_tile_size); 



template <typename Dtype>
void winoMulti_gpu(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
					const Dtype *u_matrix, Dtype *v_matrix, Dtype *m_matrix, const int wino_tile_size)
{

	int batched = (wino_tile_size + 2) * (wino_tile_size + 2); 
  //std::cout << "Matrix size (M x N) (N x K) M N K :" << num_outputs << " " << num_inputs << " "  << batchs*tileH*tileW << "\n";
  int M = num_outputs;
  int N = batchs*tileH*tileW;
  int K = num_inputs;
  dim3 gridSize((N-1)/NUM_THREADS+1, (M-1)/NUM_THREADS+1, batched);
  dim3 blockSize(NUM_THREADS, NUM_THREADS, 1);
//  std::cout << "Grid Size: " << (N-1)/NUM_THREADS + 1 << " " << (M-1)/NUM_THREADS+1 << " " << batched << "\n";
  winoMulti_gpu_kernel<Dtype><<<gridSize, blockSize>>> (u_matrix, v_matrix, m_matrix, M, N, K, 1.0, 0.0);
}

template void winoMulti_gpu<float>(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
									const float *u_matrix, float *v_matrix, float *m_matrix, const int wino_tile_size); 
template void winoMulti_gpu<double>(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
									const double *u_matrix, double *v_matrix, double *m_matrix, const int wino_tile_size); 




template <typename Dtype>
void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
				 Dtype *m_matrix, Dtype *output, const int wino_tile_size)
{
	
	int num_kernels = batchs * num_outputs * tileH * tileW;

	if(wino_tile_size == 2)
	{
		winoDst_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
					                 CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 12)
	{
		winoDstAddOpt_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 4)
	{
		wino4x4Dst_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
					                 CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 14)
	{
		wino4x4DstAddOpt_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 6)
	{
		wino6x6Dst_gpu_kernel<Dtype><<<(num_kernels + WINO6_TH-1)/WINO6_TH,
					                 WINO6_TH>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 16)
	{
		wino6x6DstAddOpt_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
					                 CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
}

template void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
						 float *m_matrix, float *output, const int wino_tile_size); 

template void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
						 double *m_matrix, double *output, const int wino_tile_size); 

template <typename Dtype> 
__global__ void trimDst_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int outH, int outW, int num_outputs, int batchs, int tNums)
{

	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (outH * outW); 
		int yIdx = (idx % (outH * outW)) / outW;
		int xIdx = idx % outW;

		if(xIdx < 0 || xIdx >= outW || yIdx < 0 || yIdx >= outH)
			return; 
		else
			dst[idx] = src[highIdx * dataH * dataW + yIdx * dataW + xIdx]; 
	}
}

template <typename Dtype>
void trimDst_gpu(const int batchs, const int num_outputs, const int c_height, const int c_width,
            const int height, const int width, const Dtype* output_pad, Dtype *output) {

	int num_kernels = batchs * num_outputs * height * width; 
	
	trimDst_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(output_pad, output, c_height, c_width, height, width, num_outputs, batchs, num_kernels);
  /*
  Dtype *out_cpu = (Dtype*)malloc(sizeof(Dtype)*height*width);
  cudaMemcpy(out_cpu, output, sizeof(Dtype)*height*width, cudaMemcpyDeviceToHost);
  std::cout << "output\n";
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << out_cpu[i*width + j] << " ";
    }
    std::cout << "\n";
  }
  free(out_cpu);*/
}
  
template void trimDst_gpu(const int batchs, const int num_outputs, const int c_height, const int c_width,
            const int height, const int width, const double* output_pad, double *output);
  
template void trimDst_gpu(const int batchs, const int num_outputs, const int c_height, const int c_width,
            const int height, const int width, const float* output_pad, float *output);

} // namespaece caffe 
