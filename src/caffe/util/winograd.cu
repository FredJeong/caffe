#include <algorithm>
#include <stdio.h>

#include "caffe/common.hpp" 
#include "caffe/util/winograd.hpp"
#include "caffe/util/math_functions.hpp"

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
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;

		//// -- project ---- ///

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
		int sIdx = batchIdx * inputs * dataW * dataH + inputIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;


		//// -- project ---- ///

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
__global__ void winoMulti_gpu_kernel(const Dtype *A, const Dtype *B, Dtype *C, int Ah, int Bw, int Aw, const float alpha, const float beta)
{
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
		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;
			
						
		//// -- project ---- //		

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

		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;


		//// -- project ---- ///

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
		wino6x6Weight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 

}

template void winoWeight_gpu<float>(const int num_inputs, const int num_outputs, 
									const float *weight, float *wino_weight, const int wino_tile_size); 
template void winoWeight_gpu<double>(const int num_inputs, const int num_outputs, 
									const double *weight, double *wino_weight, const int wino_tile_size); 




template <typename Dtype>
void padSrc_gpu(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const Dtype *input, Dtype *input_pad)
{

	int num_kernels = batchs * num_inputs * (height + height_pad * 2) * (width + width_pad * 2); 
	
	padSrc_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(input, input_pad, height, width, height + height_pad *2 , width + width_pad * 2, num_inputs, batchs, height_pad, 0, num_kernels); 

}

template void padSrc_gpu<float>(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const float *input, float *input_pad); 
template void padSrc_gpu<double>(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const double *input, double *input_pad); 


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
		wino6x6Src_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
				                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
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


	for(int i = 0; i < batched; i++)
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_outputs, batchs*tileH*tileW, num_inputs, (Dtype)1., u_matrix + i * num_inputs * num_outputs , v_matrix + i * tileH * tileW * num_inputs * batchs, (Dtype)0., m_matrix + i * batchs * num_outputs * tileH * tileW); 
	}


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
		wino6x6Dst_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
					                 CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
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

} // namespaece caffe 
