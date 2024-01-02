#include "CNN_v1.h"

#define MAXSTREAM 8

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
    // x -> số dòng
    // y -> số cột
    // x -> số mẫu
    // y -> đầu vào: kích htước -> in_rows * in_cols * in_chanels
    _foward.setSize(_input.size.x, filter.size.x * out_cols *out_rows);
    dim3 blockSize(32,32);
    dim3 gridSize((in_cols-1)/blockSize.x+1,(in_rows-1)/blockSize.y+1);
    cudaStream_t *stream = (cudaStream_t*)  malloc(sizeof(cudaStream_t)*MAXSTREAM);
    for (int i = 0 ; i < MAXSTREAM; i++)
        CHECK(cudaStreamCreate(&stream[i]));
#ifndef UnifiedMem
    double** input = (double**) malloc(sizeof(double*)*MAXSTREAM);
    double** output = (double**) malloc(sizeof(double*)*MAXSTREAM);
    double* fil;
    double* d_bias;
    
    int inbytes = sizeof(double)*in_rows*in_cols*in_channels;
    int outbytes = sizeof(double)*out_rows*out_cols*filter.size.x;
    int filterbytes = sizeof(double)*filter_width*filter_width*in_channels*filter.size.x;
    for(int i = 0 ; i < MAXSTREAM ; i ++){
      CHECK(cudaMalloc(&input[i],inbytes));
      CHECK(cudaMalloc(&output[i],outbytes));
    }
    CHECK(cudaMalloc(&fil,filterbytes));
    CHECK(cudaMalloc(&d_bias,sizeof(double)*filter.size.x));
    CHECK(cudaMemcpy(fil,this->filter.data[0],filterbytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias,bias.data[0],sizeof(double)*filter.size.x,cudaMemcpyHostToDevice));

    if (padding)
        for(int sample = 0 ; sample < _foward.size.x ; sample += MAXSTREAM){
            for(int i = 0 ; i < MAXSTREAM ; i ++){
                if (sample + i < _foward.size.x){
                CHECK(cudaMemcpyAsync(input[i],_input.data[sample+i],inbytes,cudaMemcpyHostToDevice,stream[i]));
                convPad<<<gridSize,blockSize,0,stream[i]>>>(input[i],in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output[i]);
                CHECK(cudaMemcpyAsync(_foward.data[sample+i],output[i],outbytes,cudaMemcpyDeviceToHost,stream[i]));
                }
            }
        }
    else
        for(int sample = 0 ; sample < _foward.size.x ; sample += MAXSTREAM){
            for(int i = 0 ; i < MAXSTREAM ; i ++){
                if (sample + i < _foward.size.x){
                CHECK(cudaMemcpyAsync(input[i],_input.data[sample+i],inbytes,cudaMemcpyHostToDevice,stream[i]));
                convNonePad<<<gridSize,blockSize,0,stream[i]>>>(input[i],in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output[i]);
                CHECK(cudaMemcpyAsync(_foward.data[sample+i],output[i],outbytes,cudaMemcpyDeviceToHost,stream[i]));
                }
            }
        }
    for(int i = 0 ; i < MAXSTREAM ; i ++){
      CHECK(cudaFree(input[i]));
      CHECK(cudaFree(output[i]));
    }
    free(input);
    free(output);
    cudaFree(d_bias);
    cudaFree(fil);
#else
    if (padding)
        for(int sample = 0 ; sample < _foward.size.x ; sample ++){
          for(int i = 0 ; i < MAXSTREAM ; i ++)
            if (sample + i < _foward.size.x)
              convPad<<<gridSize,blockSize,0,stream[i]>>>(_input.data[sample+i],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample+i]);
        }
    else
        for(int sample = 0 ; sample < _foward.size.x ; sample ++){
          for(int i = 0 ; i < MAXSTREAM ; i ++)
            if (sample + i < _foward.size.x)
            convNonePad<<<gridSize,blockSize,0,stream[i]>>>(_input.data[sample+i],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample+i]);
        }
#endif
  for (int i = 0 ; i < MAXSTREAM; i++)
    CHECK(cudaStreamDestroy(stream[i]));
  free(stream);
}