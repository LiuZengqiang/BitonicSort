// A Bitonic Sort program on GPU(CUDA) by LiuZengqiang
#include<cstdio>
#include<random>
#include<ctime>
#include<iostream>
#include<cuda_runtime.h>
#include<driver_types.h>

int element_num = 1024;
int thread_num_per_block = 64;

#define 
int main(){
	// 内存
	int* data_h = nullptr;
	int* data_d = nullptr;
	size_t size = element_num * sizeof(int);
	data_h = (int*)malloc(size);
	cudaError_t err = cudaSuccess;

	

	// 
	
	return 0;
}
