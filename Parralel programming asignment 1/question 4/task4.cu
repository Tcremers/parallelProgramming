#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>


__global__ void squareKernel(float* d_in, float *d_out, int threads_num){
	const unsigned int lid = threadIdx.x; // local id inside a block
	const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
	
	if(gid < threads_num){
		d_out[gid] = pow((d_in[gid]/(d_in[gid] - 2.3)),3);
	}
}

bool equal( float* h_out , float* cpu_out, int threads_num){
    for( int i = 0; i < threads_num; ++i ){
        if( h_out[i] != cpu_out[i] )
        {
            return( false );
        }
    }
    return( true );
}

void cpuMap(float* d_in, float *d_out, int threads_num){
	for(int i = 0; i < threads_num; i++){
		d_out[i] = pow((d_in[i]/(d_in[i]-2.3)),3);
	}
}


int timeval_subtract( struct timeval* result, struct timeval* t2,struct timeval* t1) {
	unsigned int resolution=1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
	(t1->tv_usec + resolution * t1->tv_sec) ;
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}



int main(int argc, char** argv){
	unsigned int num_threads = 32757;
	unsigned int mem_size = num_threads*sizeof(float);
	unsigned int block_size = 256;
	unsigned int num_blocks = ((num_threads + (block_size - 1)) / block_size);
	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	
	printf("Array size:%u\n", num_threads);
	
	// allocate host memory
	float* h_in = (float*) malloc(mem_size);
	float* h_out = (float*) malloc(mem_size);
	float* cpu_in = (float*) malloc(mem_size);
	float* cpu_out = (float*) malloc(mem_size);
	
	// initialize the memory
	for(unsigned int i=0; i<num_threads; ++i){
		h_in[i] = (float)i;
		cpu_in[i] = (float)i;
	}

	// allocate device memory
	float* d_in;
	float* d_out;
	cudaMalloc((void**)&d_in, mem_size);
	cudaMalloc((void**)&d_out, mem_size);

	// copy host memory to device
	cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
	
	

	

	// execute the kernel
	gettimeofday(&t_start, NULL);
	squareKernel<<< num_blocks, block_size>>>(d_in, d_out, num_threads);		
	gettimeofday(&t_end, NULL);
	
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
	printf("GPU took %d microseconds (%.2fms)\n",elapsed,elapsed/1000.0);
	
	// copy result from ddevice to host
	cudaMemcpy(h_out, d_out, sizeof(float)*num_threads, cudaMemcpyDeviceToHost);

	// print result
	for(unsigned int i=0; i<10; ++i) printf("%.6f - ", h_out[i]);
	printf("\n");
	
	
	// Run and time CPU function
	gettimeofday(&t_start, NULL);
	cpuMap(cpu_in, cpu_out, num_threads);
	gettimeofday(&t_end, NULL);
	
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
	printf("CPU took %d microseconds (%.2fms)\n",elapsed,elapsed/1000.0);
	
	// print result
	for(unsigned int i=0; i<10; ++i) printf("%.6f - ", cpu_out[i]);
	printf("\n");
	
	if(equal(h_out, cpu_out, num_threads)){
		printf("VALID\n");
	}
	else{
		printf("INVALID\n");
	}

	// clean-up memory
	free(h_in); free(h_out); 
	free(cpu_in); free(cpu_out);
	cudaFree(d_in); cudaFree(d_out);
}
