// A Bitonic Sort program on GPU(CUDA) by LiuZengqiang
#include<cstdio>
#include<random>
#include<ctime>
#include<iostream>
#include<cassert>
#include<cuda_runtime.h>
#include<driver_types.h>
#define checkCudaError(val) checkCuda( (val), __FILE__, __LINE__)
template<typename T>
void checkCuda(T err, const char* const file, const int line){
	if(err!=cudaSuccess){
		std::cerr<<"Error::CUDA:: "<<file<<":"<<line<<std::endl;
		exit(-1);
	}
}

int element_num = 1024;
int thread_num_per_block = 64;

int main(){
	// 内存
	int* data_h = nullptr;
	int* data_d = nullptr;
	size_t size = element_num * sizeof(int);
	data_h = (int*)malloc(size);
	checkCudaError(cudaMalloc((void**)&data_d, size));
	checkCudaError(cudaFree(data_d));
	// 
	
	return 0;
}
