#include "CNN_v1.h"

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


__global__ void convPad(double* input, int width, int height, int depth, 
		double* filter, int filterWidth, int filterdepth, double* bias, double* output){
    
    int id_x = blockIdx.x*blockDim.x + threadIdx.x;
    int id_y = blockIdx.y*blockDim.y + threadIdx.y;
    int pad = filterWidth/2;
    int col_beg = id_x-pad;
    int row_beg = id_y-pad;

    if (id_x < width && id_y < height){
        for(int ch = 0 ; ch < filterdepth ; ch ++){ // iter through number of filter
            //
            int beg_ch = ch * width * height;
            int out_idx = beg_ch + id_y * width +id_x;
            double sum = 0;
            for(int z = 0; z < depth ; z++){// iter through in_depth of image
                for(int x = 0 ; x < filterWidth ; x ++){
                    int row = row_beg + x;
                    if (row < 0)
                        row = 0;
                    if (row >= height)
                        row = height-1;
                    
                    for(int y = 0 ; y < filterWidth ; y++){
                        int col = col_beg + y;
                        if (col < 0)
                            col = 0;
                        if (col >= width)
                            col = width-1;
                        
                        sum += filter[ch*filterWidth*filterWidth*depth + z*filterWidth*filterWidth + x*filterWidth + y] * input[z*height*width + row*width + col];
                    }
                }

            }
            output[out_idx] = sum + bias[ch];
        }
    }
}


__global__ void convNonePad(double* input, int width, int height, int depth, 
		double * filter, int filterWidth,int filterdepth, double* bias, double * output){
    int id_x = blockIdx.x*blockDim.x + threadIdx.x;
    int id_y = blockIdx.y*blockDim.y + threadIdx.y;
    int pad = filterWidth-1;
    int col_beg = id_x;
    int row_beg = id_y;

    if (id_x < width - pad && id_y < height - pad){
        for(int ch = 0 ; ch < filterdepth ; ch ++){ // iter through number of filter
            //
            int beg_ch = ch * (width - pad) * (height - pad);
            int out_idx = beg_ch + row_beg * (width-pad) + col_beg;
            double sum = 0;
            for(int z = 0; z < depth ; z++){// iter through in_depth of image
                for(int x = 0 ; x < filterWidth ; x ++){
                    int row = row_beg + x;
                    if (row >= height)
                        row = height-1;
                    
                    for(int y = 0 ; y < filterWidth ; y++){
                        int col = col_beg + y;
                        if (col >= width)
                            col = width-1;
                        
                        sum += filter[ch*filterWidth*filterWidth*depth + z*filterWidth*filterWidth + x*filterWidth + y] * input[z*height*width + row*width + col];
                    }
                }

            }
            output[out_idx] = sum + bias[ch];
        }
    }
}


void CNN_cuda_v1::forward(const Matrix& _input){
    // in: _input
    // out: this->_foward;
    // filter.size.x : out_chanels;
    _foward.setSize(_input.size.x, filter.size.x * out_cols *out_rows);
    dim3 blockSize(32,32);
    dim3 gridSize((in_cols-1)/blockSize.x+1,(in_rows-1)/blockSize.y+1);

#ifndef UnifiedMem
    double* input;
    double* output;
    double* fil;
    double* d_bias;
    
    size_t inbytes = sizeof(double)*in_rows*in_cols*in_channels;
    size_t outbytes = sizeof(double)*out_rows*out_cols*filter.size.x;
    size_t filterbytes = sizeof(double)*filter_width*filter_width*in_channels*filter.size.x;
    
    CHECK(cudaMalloc(&input,inbytes));
    CHECK(cudaMalloc(&output,outbytes));
    CHECK(cudaMalloc(&fil,filterbytes));
    CHECK(cudaMalloc(&d_bias,sizeof(double)*filter.size.x));
    CHECK(cudaMemcpy(fil,this->filter.data[0],filterbytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias,bias.data[0],sizeof(double)*filter.size.x,cudaMemcpyHostToDevice));

    if (padding)
        for(int sample = 0 ; sample < _foward.size.x ; sample ++){
            CHECK(cudaMemcpy(input,_input.data[sample],inbytes,cudaMemcpyHostToDevice));
            convPad<<<gridSize,blockSize>>>(input,in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output);
            CHECK(cudaMemcpy(_foward.data[sample],output,outbytes,cudaMemcpyDeviceToHost));
        }
    else
        for(int sample = 0 ; sample < _foward.size.x ; sample ++){
            CHECK(cudaMemcpy(input,_input.data[sample],inbytes,cudaMemcpyHostToDevice));
            convNonePad<<<gridSize,blockSize>>>(input,in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output);
            CHECK(cudaMemcpy(_foward.data[sample],output,outbytes,cudaMemcpyDeviceToHost));
        }

    cudaFree(input);
    cudaFree(output);
    cudaFree(d_bias);
    cudaFree(fil);
#else
    if (padding)
        for(int sample = 0 ; sample < _foward.size.x ; sample ++){
            convPad<<<gridSize,blockSize>>>(_input.data[sample],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample]);
        }
    else
        for(int sample = 0 ; sample < _foward.size.x ; sample ++){
            convNonePad<<<gridSize,blockSize>>>(_input.data[sample],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample]);
        }
#endif
}