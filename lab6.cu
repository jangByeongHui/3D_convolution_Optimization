#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define TILE_SIZE 14
#define KERNEL_SIZE 5
#define BLOCK_SIZE (TILE_SIZE + (KERNEL_SIZE - 1))
 
// global variable, outsize any function
__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

__global__ void Convolution2D(float* d_M, float* d_N, float* d_P,int M_Width_row,int M_Width_col,int P_Width_row,int P_Width_col){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_SIZE + ty;
    int col_o = blockIdx.x * TILE_SIZE + tx;
    int row_i =row_o - ((KERNEL_SIZE - 1) / 2); // Assumes kernel size is 3
    int col_i =col_o - ((KERNEL_SIZE - 1) / 2); // Assumes kernel size is 3
    int i=0;
    int j=0;
    float output = 0.0f;
    __shared__ float Ms[TILE_SIZE+KERNEL_SIZE-1][TILE_SIZE+KERNEL_SIZE-1];

    if((row_i >= 0) && (row_i < M_Width_row) && (col_i >= 0) && (col_i < M_Width_col)){
        Ms[ty][tx] = d_M[row_i*M_Width_col + col_i];
    }  
    else{
        Ms[ty][tx] = 0.0f;
    }
    if(ty < TILE_SIZE && tx < TILE_SIZE){
        for(i = 0; i < KERNEL_SIZE; i++){
            for(j = 0; j < KERNEL_SIZE; j++){
                output += Mc[i][j] * Ms[i+ty][j+tx];
                //printf("Mc[%d][%d]:%f  Ms[%d][%d]:%f\n",i,j,Mc[i][j],i+ty,j+tx,Ms[i+ty][j+tx]);
            }
        }
        // some threads do not write output
        if(row_o < P_Width_row && col_o < P_Width_col){
            d_P[row_o * P_Width_col + col_o] = output;
        }   
    }
}
void verification(const float *N, const float *M, const float *P, int Rows, int Columns) ;
int main(int argc,char **argv){
    float *d_M,*d_N,*d_P;
    float *h_M,*h_N,*h_P;
    // cudaEvent_t start,end;
    // float time_ms=0;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);

    h_M=(float*)malloc(sizeof(float)*TILE_SIZE*TILE_SIZE );
    h_N=(float*)malloc(sizeof(float)*KERNEL_SIZE*KERNEL_SIZE);
    h_P=(float*)malloc(sizeof(float)*TILE_SIZE*TILE_SIZE);
    memset(h_P, 0,TILE_SIZE * TILE_SIZE * sizeof(float));

    for(int i=0;i<TILE_SIZE*TILE_SIZE;i++){
        h_M[i]=i+1;
    }
    for(int i=0;i<KERNEL_SIZE*KERNEL_SIZE;i++){
        h_N[i]=100+i;
    }
    
    cudaMalloc((void**)&d_M, sizeof(float) *TILE_SIZE*TILE_SIZE);
    cudaMalloc((void**)&d_N, sizeof(float) *KERNEL_SIZE*KERNEL_SIZE);
    cudaMalloc((void**)&d_P, sizeof(float) *TILE_SIZE*TILE_SIZE);

   
    cudaMemcpy(d_M,h_M,TILE_SIZE*TILE_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_P,h_P,TILE_SIZE*TILE_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mc, h_N, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE,0,cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE + (KERNEL_SIZE - 1), TILE_SIZE + (KERNEL_SIZE - 1));
    
    // cudaEventRecord(start,0);
    Convolution2D<<< 1,dimBlock>>>(d_M,d_N,d_P,TILE_SIZE,TILE_SIZE,TILE_SIZE,TILE_SIZE);
    // cudaEventRecord(end,0);
    //printf("Execution time for Cuda Convolution2D: %.2f ms \n\n",time_ms);

    cudaMemcpy(h_P,d_P,TILE_SIZE*TILE_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    
    verification(h_M, h_N, h_P, TILE_SIZE, TILE_SIZE);
    free(h_P);
    free(h_M);
    free(h_N);
    cudaFree(d_P);
    cudaFree(d_M);
    cudaFree(d_N);
   
    return 0;
}

void verification(const float *N, const float *M, const float *P, int Rows, int Columns) {
	int r, c, h, w;
	int row_i, col_i;
	bool equal;
	float* results;
	results = (float*)malloc(Rows * Columns * sizeof(float));
	memset(results, 0, Rows * Columns * sizeof(float));
  
	for (r = 0; r < Rows; r++) {
       
		for (c = 0; c < Columns; c++) {
            
			for (h = 0; h < KERNEL_SIZE; h++) {
                
				for (w = 0; w < KERNEL_SIZE; w++) {
					row_i = r - ((KERNEL_SIZE - 1) / 2) + h;
					col_i = c - ((KERNEL_SIZE - 1) / 2) + w;
					if ((row_i >= 0) && (row_i < Rows) && (col_i >= 0) && (col_i < Columns)) {
						results[r*Columns + c] += (M[h*KERNEL_SIZE + w] * N[row_i*Columns + col_i]);
					}
				}
			}
		}
	}

	equal = true;
	for (int i = 0; i < Rows * Columns && equal; i++) {
        //printf("results[%d]:%f P[%d]:%f\n",i,results[i],i,P[i]);
		if (abs(results[i] - P[i]) >= 0.001f) {
			equal = false;
			printf("NOT EQUAL!\n");
		}
	}

	if (equal) {
		printf("Results are equal!\n");
	}
	else {
		printf("Results are NOT equal!\n");
	}

	free(results);
	return;
}




#include <iostream>

#define     MASK_WIDTH      3
#define     MASK_RADIUS     MASK_WIDTH / 2
#define     TILE_WIDTH      8
#define         W           (TILE_WIDTH + MASK_WIDTH - 1)

/**
 * GPU 2D Convolution using shared memory
 */


