// widthA = heightB
template <class T> 
__global__ void matMultKer(float* A, float* B, float* C, int widthA, int heightA, int widthB) {
  T accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}


// widthA = heightB
//WIDTH_A = U, HEIGHT_A = M, WIDTH_B = N
template <class T, int TILE> 
__global__ void matMultTiledKer(float* A, float* B, float* C, int U, int M, int N) {
 
  __shared__ T Ash[TILE][TILE];
  __shared__ T Bsh[TILE][TILE];

  
  int ii = blockIdx.y * TILE; //blockDim.x==TILE
  int jj = blockIdx.x * TILE; //blockDim.y==TILE
  
  int tidy = threadIdx.y;
  int tidx = threadIdx.x;

  int i = tidy+ii;
  int j = tidx+jj;
  float accum = 0.0;
  for(int kk = 0; kk < U; kk += TILE) {
        Ash[tidy][tidx] = (i<M && kk+tidx < U ) ?
			A[i*U + (kk+tidx)] : 0.0;

		Bsh[tidy][tidx] = (j<N && kk+tidy < U ) ?
			B[(kk+tidy)*N + j] : 0.0;
		__syncthreads();
		
		for(int k=0; k<TILE; k++) {
			accum += Ash[tidy][k] * Bsh[k][tidx];
		} 
		__syncthreads();
  }
  if( (i < M) && (j < N) ){
    C[i*N + j] = accum;
	//C[j*M + i] = accum;
  }
	
}

