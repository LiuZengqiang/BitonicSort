// A Bitonic Sort program on GPU(CUDA) by LiuZengqiang on 2022.08.31
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

int n = 1<<25;
/*
 * 双调排序的主要函数
 */
__global__ void bitonicSort(int* data, int n, int len, int d, int need_thread_num){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx>=need_thread_num){
		return;
	}
	int x = idx/(len/2);	// 母双调序列下标
	int seq_left = x*len;
	int y = (idx%(len/2))/(d/2);	// 子双调序列下标
	int sub_seq_left = y*d + seq_left;
	int sub_seq_right = sub_seq_left + d;
	int z = (idx%(len/2))%(d/2);	// 在子双调序列中的下标
	int delta = z;
	// 将data[sub_seq_left, sub_seq_right]调整为升序
	if(x%2==0){
		int l = sub_seq_left + delta;
		int r = l + d/2;
		if(data[l]>data[r]){
			int temp = data[l];
			data[l] = data[r];
			data[r] = temp;
		}
	}else{
	// 将data[sub_seq_left, sub_seq_right]调整为降序
		int r = sub_seq_right - 1 - delta;
		int l = r - (d/2);
		if(data[l]<data[r]){
			int temp = data[l];
			data[l] = data[r];
			data[r] = temp;
		}
	}
}
/*
 * 检查数据是否有序(升序)
 */
bool check(int* data, int n){
	for(int i=0; i<n-1; i++){
		if(data[i]>data[i+1]){
			return false;
		}
		
	}
	return true;
}
void debug(int* data, int n){
	for(int i=0; i<n; i++){
		std::cout<<data[i]<<" ";
	}
	std::cout<<std::endl;
}

int main(){
	std::cout<<"1. 待排序数组长度为:"<<n<<"."<<std::endl;

	int* data_h = nullptr;
	int* data_d = nullptr;
	size_t size = n * sizeof(int);
	
	std::cout<<"2. 申请内存空间."<<std::endl;
	data_h = (int*)malloc(size);
	checkCudaError(cudaMalloc((void**)&data_d, size));
	
	std::cout<<"3. 使用随机数初始化数组."<<std::endl;
	std::srand(std::time(nullptr));
	for(int i=0; i<n; i++){
		data_h[i] = std::rand()%n;
	}
	
	std::cout<<"4. 将数据复制到Device端."<<std::endl;
	checkCudaError(cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice));
	
	std::cout<<"5. 开始在GPU段进行排序."<<std::endl;
	for(int len=2; len<=n; len*=2){
	// 自底向上，将长度为len的多段双调序列排好序
		for(int d=len; d>1; d/=2){
		// 自顶向下，将长度为len的多段双调序列，分解为长度为d的子u双调序列
		// 当d=2时，可以保证各段长度为len的双调序列有序(奇数段为降序，偶数段为升序)
			int need_thread_num = n/2;
			int thread_num_per_block = 64;
			int block_num_per_grid = (need_thread_num+thread_num_per_block-1)/thread_num_per_block;
			bitonicSort<<<block_num_per_grid, thread_num_per_block>>>(data_d, n, len, d, need_thread_num);
			//checkCudaError(cudaDeviceSynchronize());
		}
		checkCudaError(cudaDeviceSynchronize());
	}
	
	std::cout<<"6. 将排序好的数据复制回Host端."<<std::endl;
	checkCudaError(cudaMemcpy(data_h, data_d, size, cudaMemcpyDeviceToHost));
	
	std::cout<<"7. 检查数组是否有序:"<<std::endl;
	if(check(data_h, n)){
		std::cout<<"数组已经有序(升序.)"<<std::endl;
	}else{
		std::cout<<"数组依旧乱序."<<std::endl;
	}
	
	std::cout<<"8. 释放Host和Device端内存."<<std::endl;
	delete[] data_h;
	checkCudaError(cudaFree(data_d));
	return 0;
}
