// widthA = heightB
template <class T> 
__global__ void matTransposeKer(float* A, float* B, int rowsA, int colsA) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y; 

	if( (i < colsA) && (j < rowsA) ){
		B[j*rowsA + i] = A[i*colsA+j];
	}
}

// blockDim.y = TILE; blockDim.x = TILE
// each block transposes a square TILE
template <class T, int TILE> 
__global__ void matTransposeTiledKer(float* A, float* B, int rowsA, int colsA) {
	__shared__ T tile[TILE][TILE+1];
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int j = blockIdx.x*TILE + tidx;
	int i = blockIdx.y*TILE + tidy;
	if( j < colsA && i < rowsA )
		tile[tidy][tidx] = A[i*colsA+j];
	__syncthreads();
	i = blockIdx.y*TILE + threadIdx.x;
	j = blockIdx.x*TILE + threadIdx.y;
	if( j < colsA && i < rowsA )
		B[j*rowsA+i] = tile[tidx][tidy];
}


__global__ void 
origProg(float* A, float* B, unsigned int N) {
	const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if(gid < N){
		unsigned long long ii = gid*64;
		float tmpB = A[ii];
		tmpB = tmpB*tmpB;
		B[ii] = tmpB;
		for(int j = 1; j < 64; j++){
			float tmpA  = A[ii + j];
			float accum = sqrt(tmpB) + tmpA*tmpA;
			B[ii + j] = accum;
			tmpB = accum;
		}
	}	
}

__global__ void 
transfProg(float* A, float* B, unsigned int N) {
	const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if(gid < N){
		//B[[j*rowsA + i]] = A[i*colsA+j];
		float tmpB = A[gid];
		tmpB = tmpB*tmpB;
		B[gid] = tmpB;
		for(int j = 1; j < 64; j++){
			float tmpA  = A[j*N + gid];
			float accum = sqrt(tmpB) + tmpA*tmpA;
			B[j*N + gid] = accum;
			tmpB = accum;
		}
	}	
}

