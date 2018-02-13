#ifndef SCAN_HOST
#define SCAN_HOST

#include "ScanKernels.cu.h"

#include <sys/time.h>
#include <time.h> 

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU needs to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void scanInc(    unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out  // device
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    scanIncKernel<OP,T><<< num_blocks, block_size, sh_mem_size >>>(d_in, d_out, d_size);
    cudaThreadSynchronize();
    
    if (block_size >= d_size) { return; }

    /**********************/
    /*** Recursive Case ***/
    /**********************/

    //   1. allocate new device input & output array of size num_blocks
    T *d_rec_in, *d_rec_out;
    cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

    unsigned int num_blocks_rec = ( (num_blocks % block_size) == 0 ) ?
                                  num_blocks / block_size     :
                                  num_blocks / block_size + 1 ; 

    //   2. copy in the end-of-block results of the previous scan 
    copyEndOfBlockKernel<T><<< num_blocks_rec, block_size >>>(d_out, d_rec_in, num_blocks);
    cudaThreadSynchronize();

    //   3. scan recursively the last elements of each CUDA block
    scanInc<OP,T>( block_size, num_blocks, d_rec_in, d_rec_out );

    //   4. distribute the the corresponding element of the 
    //      recursively scanned data to all elements of the
    //      corresponding original block
    distributeEndBlock<OP,T><<< num_blocks, block_size >>>(d_rec_out, d_out, d_size);
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
}


/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates 
 *                start of a segment.
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void sgmScanInc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
    unsigned int num_blocks;
    //unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    T     *d_rec_in;
    int   *f_rec_in;
    cudaMalloc((void**)&d_rec_in, num_blocks*sizeof(T  ));
    cudaMalloc((void**)&f_rec_in, num_blocks*sizeof(int));

    sgmScanIncKernel<OP,T> <<< num_blocks, block_size, 32*block_size >>>
                    (d_in, flags, d_out, f_rec_in, d_rec_in, d_size);
    cudaThreadSynchronize();
    //cudaError_t err = cudaThreadSynchronize();
    //if( err != cudaSuccess)
    //    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

    if (block_size >= d_size) { cudaFree(d_rec_in); cudaFree(f_rec_in); return; }

    //   1. allocate new device input & output array of size num_blocks
    T   *d_rec_out;
    int *f_inds;
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T   ));
    cudaMalloc((void**)&f_inds,    d_size    *sizeof(int ));

    //   2. recursive segmented scan on the last elements of each CUDA block
    sgmScanInc<OP,T>
                ( block_size, num_blocks, d_rec_in, f_rec_in, d_rec_out );

    //   3. create an index array that is non-zero for all elements
    //      that correspond to an open segment that crosses two blocks,
    //      and different than zero otherwise. This is implemented
    //      as a CUDA-block level inclusive scan on the flag array,
    //      i.e., the segment that start the block has zero-flags,
    //      which will be preserved by the inclusive scan. 
    scanIncKernel<Add<int>,int> <<< num_blocks, block_size, flg_sh_size >>>
                ( flags, f_inds, d_size );

    //   4. finally, accumulate the recursive result of segmented scan
    //      to the elements from the first segment of each block (if 
    //      segment is open).
    sgmDistributeEndBlock <OP,T> <<< num_blocks, block_size >>>
                ( d_rec_out, d_out, f_inds, d_size );
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
    cudaFree(f_rec_in );
    cudaFree(f_inds   );
}
#endif //SCAN_HOST



template<class OP, class T>
void scanExc(const unsigned int  block_size,
			 const unsigned long d_size,
			 T*            d_in,  //device
			 T*            d_out  //device
) {
	unsigned int num_blocks;
    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;
	
	// allocate temp device output array of size d_size
	T *d_temp_out;
	cudaMalloc((void**)&d_temp_out, d_size*sizeof(T));
	
	//run incScan
	scanInc<OP,T>( block_size, d_size, d_in, d_temp_out);
	cudaThreadSynchronize();
	// call shiftRightByOne for temp array
	shiftRightByOne<T> <<<num_blocks, block_size>>>(d_temp_out, d_out, OP::identity(), d_size);
	
	// clean up
	cudaFree(d_temp_out );
}



template<class OP, class T>
void sgmScanExc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
	unsigned int num_blocks;
    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;
	
	// allocate temp device output array of size d_size
	T *d_temp_out;
	cudaMalloc((void**)&d_temp_out, d_size*sizeof(T));
	
	//run incScan
	sgmScanInc<OP,T> ( block_size, d_size, d_in, flags, d_temp_out );
	cudaThreadSynchronize();
	// call shiftRightByOne for temp array
	sgmShiftRightByOne<T> <<<num_blocks, block_size>>>(d_temp_out, flags, d_out, OP::identity(), d_size);
	
	// clean up
	cudaFree(d_temp_out );
}


/**
* This implements the whole bunch of MSSP:
 * inp_d    the original array (of ints)
 * inp_lift the result array
 * inp_size is the size of the original (and output) array
 *              in number of int (MyInt4) elements
 **/

void mssp(const unsigned int  block_size,
		 const unsigned long inp_size,
		 int* 	 inp_d, 
		 MyInt4* inp_lift
){
	unsigned int num_blocks;
    num_blocks = ( (inp_size % block_size) == 0) ?
                    inp_size / block_size     :
                    inp_size / block_size + 1 ;
					
	//Create temp output array as input for reduce.
	MyInt4 *temp_inp_lift;
	cudaMalloc((void**)&temp_inp_lift, inp_size*sizeof(MyInt4));
	
	msspTrivialMap<<< num_blocks, block_size>>>(inp_d, temp_inp_lift, inp_size);
	cudaThreadSynchronize();
	
	scanInc<MsspOp,MyInt4>( block_size, inp_size, temp_inp_lift, inp_lift);
	
}

/**
 * This implements the whole bunch of sparce matric-vector multiplication:
 * block_size	the size of the cuda blucks
 * d_shp		the shape of the sparce matrix, array with number of non-zero elements per row
 * d_num_elms	the total ammount of non zero elements in the sparce matrix
 * d_num_rows	the number of rows in the matrix
 * d_mat_inds	the column indices of the matrix values
 * d_mat_vals	the non zero matrix values
 * d_vect		the vector values
 * d_res		the result array
 **/
void spMVmult(const unsigned int block_size,
			 const unsigned int d_num_elms,
			 const unsigned int d_num_rows,
			 int* d_shp,
			 int* d_mat_inds,
			 float* d_mat_vals,
			 float* d_vect,
			 float* d_res
){
	//Number of blocks for kernels over total matrix
	unsigned int num_blocks_tot;
	num_blocks_tot = ( (d_num_elms % block_size) == 0) ?
                    d_num_elms / block_size     :
                    d_num_elms / block_size + 1 ;
					
	//Number of blocks for kernels over num_rows
	unsigned int num_blocks_rows;
	num_blocks_rows = ( (d_num_rows % block_size) == 0) ?
                    d_num_rows / block_size     :
                    d_num_rows / block_size + 1 ;
	
	//I'm copying how to make the flag array from spMVmult-flat.fut
	
	//Allocate flag arrays and tmp_pairs array
	int *d_rowIota;
	int *d_shp_sc;
	int *d_shp_inds;
	int *d_flags;
	int *d_tmp_inds;
	
	float *d_tmp_pairs;
	
	cudaMalloc((void**)&d_rowIota, d_num_rows*sizeof(int));
	cudaMalloc((void**)&d_shp_sc, d_num_rows*sizeof(int));
	cudaMalloc((void**)&d_shp_inds, d_num_rows*sizeof(int));
	
	cudaMalloc((void**)&d_flags, d_num_elms*sizeof(int));
	cudaMalloc((void**)&d_tmp_inds, d_num_elms*sizeof(int));
	cudaMalloc((void**)&d_tmp_pairs, d_num_elms*sizeof(float));
	
	//Call kernels to make flags array
	iota<int><<< num_blocks_rows, block_size>>>(d_num_rows, d_rowIota);
	replicate<int><<< num_blocks_tot, block_size >>>(d_num_elms, 0, d_flags);
	
	scanInc< Add<int>, int>( block_size, d_num_rows, d_shp, d_shp_sc );
	mapShpInds<<< num_blocks_rows, block_size>>>(d_num_rows, d_rowIota, d_shp_sc, d_shp_inds);
	scatterFlags<<< num_blocks_rows, block_size>>>(d_num_rows, d_shp_inds, d_flags);
	scanInc< Add<int>, int>( block_size, d_num_elms, d_flags, d_tmp_inds );
	
	//Apply mat-vct multiplaction
	spMatVctMult_pairs<<< num_blocks_tot, block_size>>>(d_mat_inds, d_mat_vals, d_vect, d_num_elms, d_tmp_pairs);
	
	write_lastSgmElem<<< num_blocks_tot, block_size>>>(d_tmp_pairs, d_tmp_inds, d_flags, d_num_elms, d_res);
	
	
	
	//clean up
	cudaFree(d_rowIota);
	cudaFree(d_shp_sc);
	cudaFree(d_shp_inds);
	cudaFree(d_flags);
	cudaFree(d_tmp_inds);
	cudaFree(d_tmp_pairs);
}

























