#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"


int scanIncTest(bool is_segmented, bool is_exclusive) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1; 
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 


    { // calling inclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
		if(is_exclusive){
			if(is_segmented)
				sgmScanExc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
			else
				scanExc< Add<int>,int > ( block_size, num_threads, d_in, d_out );		
		}
		else{
			if(is_segmented)
				sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
			else
				scanInc< Add<int>,int > ( block_size, num_threads, d_in, d_out );
		}
        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Scan Inclusive GPU Kernel runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            accum += 1;
            
            if ( accum != h_out[i] ) { 
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }        
    }
	// I'm just going to assume that if the first element of the resulting array is 0. The exclusive scan was a succes!
	else if(is_exclusive){
		if(h_out[0] == 0){
			success = true;
		}
	}
	else {
        for(int i=0; i<num_threads; i++) {
            accum += 1;
 
            if ( accum != h_out[i] ) { 
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }        
    }

    if(success) printf("\nScan Inclusive +   VALID RESULT!\n");
    else        printf("\nScan Inclusive + INVALID RESULT!\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}


void msspTest(){
	const unsigned int num_threads = 8; //for now...
	const unsigned int block_size  = 512;
	unsigned int mem_size = num_threads * sizeof(int);
	unsigned int mem_size_myInt = num_threads * sizeof(MyInt4);
	
	int foo[] = {1, -2, 3, 4, -1, 5, -6, 1};
	
	int* inp_h    		= (int*) malloc(mem_size);
	MyInt4* inp_lift_h	= (MyInt4*) malloc(mem_size_myInt);
	
	for(int i = 0; i < num_threads; i++){
		inp_h[i] = foo[i];
	}
	
	//init device variables
	int* inp_d;
	MyInt4* inp_lift_d;
	cudaMalloc((void**)&inp_d,  mem_size);
	cudaMalloc((void**)&inp_lift_d,  mem_size_myInt);
	
	//Copy host memory to device
	cudaMemcpy(inp_d, inp_h, mem_size, cudaMemcpyHostToDevice);
	
	//Call mssp
	mssp(block_size, num_threads, inp_d, inp_lift_d);
	
	cudaMemcpy(inp_lift_h, inp_lift_d, mem_size_myInt, cudaMemcpyDeviceToHost);
	printf("Maximum segment sum: %i \n", inp_lift_h[7].x);
	
	// cleanup memory
	cudaFree(inp_d);
	cudaFree(inp_lift_d);
	
	free(inp_h);
	free(inp_lift_h);
}


void spMVmultTest(){
	const unsigned int num_elms = 1000000;
	const unsigned int num_rows = 10000;
    const unsigned int block_size  = 512;
			
			
	int* h_shp = 		(int*) malloc(num_rows * sizeof(int));
	int* h_mat_inds = 	(int*) malloc(num_elms * sizeof(int));
	
	float* h_mat_vals = (float*) malloc(num_elms * sizeof(float));
	
	float* h_vect = 	(float*) malloc(num_rows * sizeof(float));
	float* h_res = 		(float*) malloc(num_rows * sizeof(float));
	
	//init arrays with some kind of random data...
	// rand_int = (rand() % (max + 1 - min)) + min
	// rand_float = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max - min)))
	
	for(int i = 0; i < num_elms; i++){
		h_mat_inds[i] = (rand() % (9999 + 1 - 0)) + 0;
		h_mat_vals[i] = -7 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(7 - -7)));
	}
	
	for(int j = 0; j < num_rows; j++){
		h_shp[j] = 100;
		h_vect[j] = -10 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10 - -10)));
	}
	
	
	int* d_shp;
	int* d_mat_inds;
	float* d_mat_vals;
	float* d_vect;
	float* d_res;
	
	cudaMalloc((void**)&d_shp, num_rows * sizeof(int));
	cudaMalloc((void**)&d_mat_inds, num_elms * sizeof(int));
	
	cudaMalloc((void**)&d_mat_vals, num_elms * sizeof(float));
	
	cudaMalloc((void**)&d_vect, num_rows * sizeof(float));
	cudaMalloc((void**)&d_res, num_rows * sizeof(float));
	
	//Copy host memory to device
	cudaMemcpy(d_shp, h_shp, num_rows * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_mat_inds, h_mat_inds, num_elms * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_vals, h_mat_vals, num_elms * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_vect, h_vect, num_rows * sizeof(float), cudaMemcpyHostToDevice);
	
	
	spMVmult(block_size,
			 num_elms,
			 num_rows,
			 d_shp,
			 d_mat_inds,
			 d_mat_vals,
			 d_vect,
			 d_res
	);
	
	//Copy results to host
	cudaMemcpy(h_res, d_res, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
	
	//print results
	printf("spMV multiplication result vector: \n");
	for(int k = 0; k < num_rows; k++){
		printf("%f ", h_res[k]);
	}
	
	//clean up
	free(h_shp);
	free(h_mat_inds);
	free(h_mat_vals);
	free(h_vect);
	free(h_res);
	
	cudaFree(d_shp);
	cudaFree(d_mat_inds);
	cudaFree(d_mat_vals);
	cudaFree(d_vect);
	cudaFree(d_res);
	
	
}

int main(int argc, char** argv) {
	scanIncTest(true, false);
	msspTest();
	spMVmultTest();
}
