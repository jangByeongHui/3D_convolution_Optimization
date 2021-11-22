#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <unistd.h>

#define MASK_WIDTH 5
#define TILE_SIZE 4


__constant__ float Mc[1024];

__global__ void Convolution3d(float* input, float* output, int numARows, int numACols, int numAHeight, int numCRows, int numCCols, int numCHeight, int k_dim)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_o = blockIdx.y*TILE_SIZE + ty;
    int col_o = blockIdx.x*TILE_SIZE + tx;
    int height_o = blockIdx.z*TILE_SIZE + tz;
    
    int row_i = row_o - ((k_dim -1)/2);
    int col_i = col_o - ((k_dim -1)/2);
    int height_i = height_o - ((k_dim -1)/2);

    __shared__ float Ns[1024];
    
    if((row_i >= 0) && (row_i < numARows) && (col_i >= 0) && (col_i < numACols) && (height_i >= 0) && (height_i < numAHeight)){
        Ns[tz*(TILE_SIZE + (k_dim - 1))*(TILE_SIZE + (k_dim - 1)) + ty*(TILE_SIZE + (k_dim - 1)) + tx] = input[numACols*numARows*height_i + row_i*numACols+col_i];
    }else{
        Ns[tz*(TILE_SIZE + (k_dim - 1))*(TILE_SIZE + (k_dim - 1)) + ty*(TILE_SIZE + (k_dim - 1)) + tx] = 0.0f;
    }
    __syncthreads();
    if(ty < TILE_SIZE && tx < TILE_SIZE && tz < TILE_SIZE){
        float out = 0.0f;
        for(int i = 0; i < k_dim; i++){
            for(int j = 0; j < k_dim; j++){
                for(int k = 0; k < k_dim; k++){
                    out += Mc[i*k_dim*k_dim+j*k_dim+k] * Ns[(i+tz)*(TILE_SIZE + (k_dim - 1))*(TILE_SIZE + (k_dim - 1)) + (j+ty)*(TILE_SIZE + (k_dim - 1)) + k+tx];
                }
            }
        }
        __syncthreads();
        if(row_o < numCRows && col_o < numCCols && height_o < numCHeight){
            output[height_o * numCCols * numCRows + row_o * numCCols + col_o] = out;
        }
    }
}


int main(int argc, char** argv)
{
    int check;
    extern char *optarg;
	extern int optind;
    check = getopt(argc, argv, "abcde :");
    FILE* input_file;
    FILE* output_file;
    FILE* kernel_file;
    float *input,*output,*kernel;
    float *d_input;
    switch (check)
    {
        case 'a':
            input_file = fopen("./sample/test1/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test1/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test1/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'b':
            input_file = fopen("./sample/test2/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test2/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test2/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'c':
            input_file = fopen("./sample/test3/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test3/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test3/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'd':
            input_file = fopen("./sample/test4/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test4/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test4/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'e':
            input_file = fopen("./sample/test5/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test5/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test5/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        default:
            printf("Wrong argument ./conv3d -[a...e]");
            break;
    }

    int i_x,i_y,i_z;
    int o_x,o_y,o_z;
    int k_dim;
    
    fscanf(input_file,"%d %d %d",&i_z,&i_y,&i_x);
    input=(float*)malloc(sizeof(float)*i_x*i_y*i_z);
    cudaMalloc(&d_input,sizeof(float)*i_x*i_y*i_z);
    
    for(int i=0;i<i_x*i_y*i_z;i++){
        fscanf(input_file,"%f",input+i);
    }
    cudaMemcpy(d_input,input, sizeof(float)*i_x*i_y*i_z, cudaMemcpyHostToDevice);

    fscanf(kernel_file,"%d",&k_dim);
    kernel=(float*)malloc(sizeof(float)*k_dim*k_dim*k_dim);
    for(int i=0;i<k_dim*k_dim*k_dim;i++){
        fscanf(kernel_file,"%f",kernel+i);
    }
    cudaMemcpyToSymbol(Mc,kernel, sizeof(float)*k_dim*k_dim*k_dim);

    fscanf(output_file,"%d %d %d",&o_z,&o_y,&o_x);
    output=(float*)malloc(sizeof(float)*o_x*o_y*o_z);
    for(int i=0;i<o_x*o_y*o_z;i++){
        fscanf(output_file,"%f",output+i);
    }

    printf("input: %d %d %d \n", i_x, i_y, i_z);
    printf("kernel: %d\n", k_dim);
    printf("output: %d %d %d\n", o_x, o_y, o_z);

    float *h_out, *d_out;
    h_out = (float*)malloc(sizeof(float)*o_x*o_y*o_z);
    cudaMalloc(&d_out, sizeof(float)*o_x*o_y*o_z);

    int blocksize = TILE_SIZE + (k_dim - 1);
    dim3 dimGrid(ceil(i_x/(TILE_SIZE*1.0)), ceil(i_y/(TILE_SIZE*1.0)), ceil(i_z/(TILE_SIZE*1.0)));
    dim3 dimBlock(blocksize, blocksize, blocksize);
    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
	cudaEventCreate(&end);
    
    cudaEventRecord(start);
    Convolution3d<<<dimGrid, dimBlock>>>(d_input, d_out, i_y, i_x, i_z, o_y, o_x, o_z, k_dim);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("Execution time for CUDA: %.3fms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    //cudaEventRecord(end,0);
	//cudaEventSynchronize(end);
	//cudaEventElapsedTime(&time_ms, start, end);


    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float)*o_x*o_y*o_z, cudaMemcpyDeviceToHost);

    /*
    for(int i = 0; i < o_x*o_y*o_z; i++){
        if(i % (o_x*o_y) == 0){
            printf("\n");
        }
        if(i % o_x == 0){
            printf("\n");
        }
        printf("%f ", h_out[i]);
    }
    */

    int err = 0;
    for(int i=0;i<o_x*o_y*o_z;i++){
        if(abs(h_out[i] - output[i]) >= 0.001f){
            err++;
        }
    }
    
    if(err == 0){
        printf("validation complete\n");
    }else{
        printf("%d\n", err);
    }
    
    

	return EXIT_SUCCESS;
}