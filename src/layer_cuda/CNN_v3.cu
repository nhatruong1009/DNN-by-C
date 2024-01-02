#include "CNN_v3.h"
#define TILE_WIDTH 32

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}



__global__ void matrix_multiplication_kernel_withBias(double* A, double* B, double* C, int m, int n, int k, double *bias , int chanel_size)
{
	__shared__ double s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ double s_B[TILE_WIDTH][TILE_WIDTH];
	//TODO
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    
    double Cvalue = bias[Col/chanel_size];
    for (int ph = 0; ph < ((n - 1) / TILE_WIDTH) + 1; ph++) {
        // copy data
        if (threadIdx.x + (ph * TILE_WIDTH) < n && Row < m) {
            s_A[threadIdx.y][threadIdx.x] = A[(Row * n) + threadIdx.x + (ph * TILE_WIDTH)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (threadIdx.y + ph * TILE_WIDTH < n && Col < k) {
            s_B[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * k + Col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();


        // cal submatrix
        for (int j = 0; j < TILE_WIDTH; ++j) {
            Cvalue += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        }
         __syncthreads();
      }
    if (Row < m && Col < k)
      C[Row * k + Col] = Cvalue;

}

__global__ void TransposeKernel(const double * input, double *output, int in_r, int in_c){
    __shared__ double s_mem[32][33];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < in_c && idy < in_r)
        s_mem[threadIdx.x][threadIdx.y] = input[idy*in_c + idx];
    __syncthreads();

    idx = blockIdx.y * blockDim.y + threadIdx.x;
    idy = blockIdx.x * blockDim.x + threadIdx.y;

    if (idx < in_r & idy < in_c)
        output[idy*in_r + idx] = s_mem[threadIdx.y][threadIdx.x];
    
}

void CNN_cuda_v3::TransformFillter(double* filter_out,int n, int in_height,int in_width){
    double* d_temp_filter;
    CHECK(cudaMalloc(&d_temp_filter,sizeof(double)*n));
    int n_cpy = n -  (in_height - filter_width) * in_width - (in_width - filter_width);
    for (int ch = 0 ; ch < filter.size.x ; ch ++){
        CHECK(cudaMemset(d_temp_filter,0,sizeof(double)*n));
        for(int i = 0 ; i < in_channels ; i++){
            for(int r = 0 ; r < filter_width ; r ++){
                CHECK(cudaMemcpy(&d_temp_filter[i*in_height*in_width + in_width*r ],&filter[ch][i*filter_width*filter_width + r*filter_width],sizeof(double)*filter_width,cudaMemcpyHostToDevice));
            }
        }

        for(int out_r = 0 ; out_r < out_rows ; out_r ++){
            for (int out_c = 0 ; out_c < out_cols ; out_c ++){
                CHECK(cudaMemcpy(&filter_out[ (ch*out_rows*out_cols + out_r*out_cols + out_c)*n + out_r * in_width + out_c ],d_temp_filter,sizeof(double)*n_cpy,cudaMemcpyDeviceToDevice));
            }
        }
    }
    cudaFree(d_temp_filter);
}

void CNN_cuda_v3::forward(const Matrix& _input){

    int m = out_rows*out_cols*filter.size.x;
    int n = in_rows*in_cols*in_channels;
    if (padding)
        n = (in_rows + filter_width - 1) * (in_cols + filter_width -1) * in_channels;
    int k = _input.size.x;

    int in_width = in_cols + ( filter_width - 1) * padding;
    int in_height = in_rows + ( filter_width - 1) * padding;
    dim3 blockSize(32,32);
    dim3 gridSize((m-1)/blockSize.x+1,(k-1)/blockSize.y+1);

#ifndef UnifiedMem
    double *d_filter;
    double *d_input;
    double *d_output;
    double *d_bias;
    
    CHECK(cudaMalloc(&d_filter,sizeof(double)*m*n));
    CHECK(cudaMemset(d_filter,0,sizeof(double)*m*n));
    //transform filter
    TransformFillter(d_filter,n,in_height,in_width);
    double * filter_transpose;
    CHECK(cudaMalloc(&filter_transpose,sizeof(double)*m*n));
    dim3 gridFilter((n-1)/blockSize.x+1,(m-1)/blockSize.y+1);
    TransposeKernel<<<gridFilter,blockSize>>>(d_filter,filter_transpose,m,n);
    cudaFree(d_filter);
    
    CHECK(cudaMalloc(&d_bias,sizeof(double)*bias.size.y));
    CHECK(cudaMemcpy(d_bias,bias.data[0],sizeof(double)*bias.size.y,cudaMemcpyHostToDevice));

    // input
    CHECK(cudaMalloc(&d_input,sizeof(double)*n*k));
    if (!padding){
        if(_input.is_one_block()){
            CHECK(cudaMemcpy(d_input,_input.data[0],sizeof(double)*n*k,cudaMemcpyHostToDevice));
        }
        else{
            for(int i = 0 ; i < k ; i ++){
                CHECK(cudaMemcpy(&d_input[i*n],_input.data[i],sizeof(double)*n,cudaMemcpyHostToDevice));
            }
        }
    }
    else{
        CHECK(cudaMemset(d_input,0,sizeof(double)*k*n));
        int pad = filter_width/2;
        int sizecpy = in_cols*sizeof(double);

        for(int sample = 0; sample < _input.size.x;sample ++){
            for(int depth = 0 ; depth < in_channels; depth++){
                for (int r = 0; r < in_rows; r++){
                    CHECK(cudaMemcpy(&d_input[sample*in_width*in_height*in_channels + depth*in_width*in_height + (r + pad)*in_width + pad],&_input.data[sample][depth*in_rows*in_cols + r*in_cols],sizecpy,cudaMemcpyHostToDevice));
                }        
            }
        }
        // data in padding
    }

    CHECK(cudaMalloc(&d_output,sizeof(double)*m*k));
    matrix_multiplication_kernel_withBias<<<gridSize,blockSize>>>(d_input,filter_transpose,d_output,k,n,m,d_bias,out_cols*out_rows);
    _foward.setSize(_input.size.x, filter.size.x * out_cols *out_rows);
    CHECK(cudaMemcpy(_foward.data[0],d_output,sizeof(double)*m*k,cudaMemcpyDeviceToHost));    
    
    cudaFree(filter_transpose);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bias);
#else
    double *d_filter;
    CHECK(cudaMalloc(&d_filter,sizeof(double)*m*n));
    CHECK(cudaMemset(d_filter,0,sizeof(double)*m*n));
    TransformFillter(d_filter,n,in_height,in_width);

    double * filter_transpose;
    CHECK(cudaMalloc(&filter_transpose,sizeof(double)*m*n));
    dim3 gridFilter((n-1)/blockSize.x+1,(m-1)/blockSize.y+1);
    TransposeKernel<<<gridFilter,blockSize>>>(d_filter,filter_transpose,m,n);
    CHECK(cudaFree(d_filter));

    _foward.setSize(_input.size.x, filter.size.x * out_cols *out_rows);
    if(!padding){
        if (_input.is_one_block())
            matrix_multiplication_kernel_withBias<<<gridSize,blockSize>>>(_input.data[0],filter_transpose,_foward.data[0],k,n,m,bias.data[0],out_cols*out_rows);
        else{
            double *d_input;
            CHECK(cudaMalloc(&d_input,sizeof(double)*n*k));
            for(int i = 0 ; i < k ; i ++)
                CHECK(cudaMemcpy(&d_input[i*n],_input.data[i],sizeof(double)*n,cudaMemcpyHostToDevice));
            matrix_multiplication_kernel_withBias<<<gridSize,blockSize>>>(d_input,filter_transpose,_foward.data[0],k,n,m,bias.data[0],out_cols*out_rows);
            cudaFree(d_input);
        }
    }
    else{
        double*d_input;
        CHECK(cudaMemset(d_input,0,sizeof(double)*k*n));
        int pad = filter_width/2;
        int sizecpy = in_cols*sizeof(double);

        for(int sample = 0; sample < _input.size.x;sample ++){
            for(int depth = 0 ; depth < in_channels; depth++){
                for (int r = 0; r < in_rows; r++){
                    CHECK(cudaMemcpy(&d_input[sample*in_width*in_height*in_channels + depth*in_width*in_height + (r + pad)*in_width + pad],&_input.data[sample][depth*in_rows*in_cols + r*in_cols],sizecpy,cudaMemcpyHostToDevice));
                }        
            }
        }
        // data in padding

        //
        matrix_multiplication_kernel_withBias<<<gridSize,blockSize>>>(d_input,filter_transpose,_foward.data[0],k,n,m,bias.data[0],out_cols*out_rows);
        cudaFree(d_input);
    }
    cudaFree(filter_transpose);

#endif
}