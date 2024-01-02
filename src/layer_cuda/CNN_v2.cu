#include "CNN_v2.h"

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
    double* filter, int filterWidth, int filterdepth, double* bias, double* output, int shared_width, int shared_height){

    extern __shared__ double s_input[]; // inputs need: (block_width(32) + pad -1) * (height_width(32) + pad - 1) 
        
    int id_x = blockIdx.x*blockDim.x + threadIdx.x; // col
    int id_y = blockIdx.y*blockDim.y + threadIdx.y; // row
    int pad = filterWidth/2;
    int border = filterWidth - 1;
    //copy data to s_input;
    int col_beg = id_x - pad;
    int row_beg = id_y - pad;
    {
        int row = row_beg;
        int col = col_beg;
        if (row < 0)
            row = 0;
        else if (row >= height)
            row = height - 1;

        if (col < 0)
            col = 0;
        else if (col >= width)
            col = width - 1;
        

        for(int d = 0 ; d < depth ; d++)
            s_input[ d * shared_width * shared_height + threadIdx.y * shared_width + threadIdx.x] = input[ d * width * height + row * width + col];

        if(threadIdx.y < border){
            int t_row = row_beg + blockDim.x;
            if (t_row >= height)
                t_row = height - 1;
            for(int d = 0 ; d < depth ; d++)
                s_input[ d * shared_width * shared_height + (threadIdx.y + blockDim.y) * shared_width + threadIdx.x] = input[ d * width * height + t_row * width + col];
        }
        if (threadIdx.x < border){
            int t_col = col_beg + blockDim.x;
            if (t_col >= width)
                t_col = width - 1;
            for(int d = 0 ; d < depth ; d++)
                s_input[ d * shared_width * shared_height + threadIdx.y * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + row * width + t_col];
        }
        if (threadIdx.x < border && threadIdx.y < border){
            int t_row = row_beg + blockDim.x;
            if (t_row >= height)
                t_row = height - 1;
            int t_col = col_beg + blockDim.x;
            if (t_col >= width)
                t_col = width - 1;
            for(int d = 0 ; d < depth ; d++)
                s_input[ d * shared_width * shared_height + (threadIdx.y + blockDim.y ) * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + t_row * width + t_col];
        }
    }
    __syncthreads();
    // calculate;

    if(id_x < width && id_y < height){

        for(int ch = 0 ; ch < filterdepth ; ch ++){ // iter through number of filter
            //
            int out_idx = ch * width * height + id_y * width +id_x;
            double sum = 0;
            for(int z = 0; z < depth ; z++){// iter through in_depth of image
                for(int x = 0 ; x < filterWidth ; x ++){
                    int row = threadIdx.y + x;
                    for(int y = 0 ; y < filterWidth ; y++){
                        int col = threadIdx.x + y;                        
                        sum += filter[ch*filterWidth*filterWidth*depth + z*filterWidth*filterWidth + x*filterWidth + y] * 
                            s_input[z*shared_height*shared_width + row*shared_width + col];
                    }
                }

            }
            output[out_idx] = sum + bias[ch];
        }
    }
}

__global__ void convNonePad(double* input, int width, int height, int depth,     
    double* filter, int filterWidth, int filterdepth, double* bias, double* output, int shared_width, int shared_height){

    extern __shared__ double s_input[]; // inputs need: (block_width(32) + pad -1) * (height_width(32) + pad - 1) 
        
    int id_x = blockIdx.x*blockDim.x + threadIdx.x; // col
    int id_y = blockIdx.y*blockDim.y + threadIdx.y; // row
    int border = filterWidth - 1;
    //copy data to s_input;
    int col_beg = id_x;
    int row_beg = id_y;
    {
        int row = row_beg;
        int col = col_beg;
        if (row >= height)
            row = height - 1;
        if (col >= width)
            col = width - 1;
        

        for(int d = 0 ; d < depth ; d++)
            s_input[ d * shared_width * shared_height + threadIdx.y * shared_width + threadIdx.x] = input[ d * width * height + row * width + col];

        if(threadIdx.y < border){
            int t_row = row_beg + blockDim.x;
            if (t_row >= height)
                t_row = height - 1;
            for(int d = 0 ; d < depth ; d++)
                s_input[ d * shared_width * shared_height + (threadIdx.y + blockDim.y) * shared_width + threadIdx.x] = input[ d * width * height + t_row * width + col];
        }
        if (threadIdx.x < border){
            int t_col = col_beg + blockDim.x;
            if (t_col >= width)
                t_col = width - 1;
            for(int d = 0 ; d < depth ; d++)
                s_input[ d * shared_width * shared_height + threadIdx.y * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + row * width + t_col];
        }
        if (threadIdx.x < border && threadIdx.y < border){
            int t_row = row_beg + blockDim.x;
            if (t_row >= height)
                t_row = height - 1;
            int t_col = col_beg + blockDim.x;
            if (t_col >= width)
                t_col = width - 1;
            for(int d = 0 ; d < depth ; d++)
                s_input[ d * shared_width * shared_height + (threadIdx.y + blockDim.y ) * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + t_row * width + t_col];
        }
    }
    __syncthreads();
    // calculate;

    int out_width = width - filterWidth + 1;
    int out_height = height - filterWidth + 1;
    if(id_x < width - filterWidth + 1 && id_y < height - filterWidth + 1){

        for(int ch = 0 ; ch < filterdepth ; ch ++){ // iter through number of filter
            //
            int out_idx = ch * out_width * out_height + id_y * out_width +id_x;
            double sum = 0;
            for(int z = 0; z < depth ; z++){// iter through in_depth of image
                for(int x = 0 ; x < filterWidth ; x ++){
                    int row = threadIdx.y + x;
                    for(int y = 0 ; y < filterWidth ; y++){
                        int col = threadIdx.x + y;                        
                        sum += filter[ch*filterWidth*filterWidth*depth + z*filterWidth*filterWidth + x*filterWidth + y] * 
                            s_input[z*shared_height*shared_width + row*shared_width + col];
                    }
                }

            }
            output[out_idx] = sum + bias[ch];
        }
    }
}


__global__ void convNonePad1_shared(double* input, int width, int height, int depth,     
    double* filter, int filterWidth, int filterdepth, double* bias, double* output, int shared_width, int shared_height){

    extern __shared__ double s_input[]; // inputs need: (block_width(32) + pad -1) * (height_width(32) + pad - 1) 
        
    int id_x = blockIdx.x*blockDim.x + threadIdx.x; // col
    int id_y = blockIdx.y*blockDim.y + threadIdx.y; // row
    int col_beg = id_x;
    int row_beg = id_y;
    int border = filterWidth - 1;
    for(int d = 0 ; d < depth ; d++){//repeat multiple times;
        //copy data;
        {
            int row = row_beg;
            int col = col_beg;
            if (row >= height)
                row = height - 1;
            if (col >= width)
                col = width - 1;

            s_input[threadIdx.y * shared_width + threadIdx.x] = input[ d * width * height + row * width + col];

            if(threadIdx.y < border){
                int t_row = row_beg + blockDim.x;
                if (t_row >= height)
                    t_row = height - 1;
                s_input[(threadIdx.y + blockDim.y) * shared_width + threadIdx.x] = input[ d * width * height + t_row * width + col];
            }
            if (threadIdx.x < border){
                int t_col = col_beg + blockDim.x;
                if (t_col >= width)
                    t_col = width - 1;
                s_input[threadIdx.y * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + row * width + t_col];
            }
            if (threadIdx.x < border && threadIdx.y < border){
                int t_row = row_beg + blockDim.x;
                if (t_row >= height)
                    t_row = height - 1;
                int t_col = col_beg + blockDim.x;
                if (t_col >= width)
                    t_col = width - 1;
                s_input[(threadIdx.y + blockDim.y ) * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + t_row * width + t_col];
            }
        }

        __syncthreads();

        int out_width = width - filterWidth + 1;
        int out_height = height - filterWidth + 1;
        if(id_x < width - filterWidth + 1 && id_y < height - filterWidth + 1){

        for(int ch = 0 ; ch < filterdepth ; ch ++){ // iter through number of filter
            //
            int out_idx = ch * out_width * out_height + id_y * out_width +id_x;
            double sum = 0;
            for(int x = 0 ; x < filterWidth ; x ++){
                int row = threadIdx.y + x;
                for(int y = 0 ; y < filterWidth ; y++){
                    int col = threadIdx.x + y;                        
                    sum += filter[ch*filterWidth*filterWidth*depth + d*filterWidth*filterWidth + x*filterWidth + y] * 
                        s_input[row*shared_width + col];
                }
            }
            if(d == 0)
                output[out_idx] = sum + bias[ch];
            else
                output[out_idx] +=sum;
        }
        }
         __syncthreads();

    }
}


__global__ void convPad1_shared(double* input, int width, int height, int depth,     
    double* filter, int filterWidth, int filterdepth, double* bias, double* output, int shared_width, int shared_height){

    extern __shared__ double s_input[]; // inputs need: (block_width(32) + pad -1) * (height_width(32) + pad - 1) 
        
    int id_x = blockIdx.x*blockDim.x + threadIdx.x; // col
    int id_y = blockIdx.y*blockDim.y + threadIdx.y; // row
    int pad = filterWidth/2;
    int col_beg = id_x-pad;
    int row_beg = id_y-pad;
    int border = filterWidth - 1;
    for(int d = 0 ; d < depth ; d++){//repeat multiple times;
        //copy data;
    {
        int row = row_beg;
        int col = col_beg;
        if (row < 0)
            row = 0;
        else if (row >= height)
            row = height - 1;

        if (col < 0)
            col = 0;
        else if (col >= width)
            col = width - 1;
        
        s_input[threadIdx.y * shared_width + threadIdx.x] = input[ d * width * height + row * width + col];

        if(threadIdx.y < border){
            int t_row = row_beg + blockDim.x;
            if (t_row >= height)
                t_row = height - 1;
            s_input[(threadIdx.y + blockDim.y) * shared_width + threadIdx.x] = input[ d * width * height + t_row * width + col];
        }
        if (threadIdx.x < border){
            int t_col = col_beg + blockDim.x;
            if (t_col >= width)
                t_col = width - 1;
            s_input[threadIdx.y * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + row * width + t_col];
        }
        if (threadIdx.x < border && threadIdx.y < border){
            int t_row = row_beg + blockDim.x;
            if (t_row >= height)
                t_row = height - 1;
            int t_col = col_beg + blockDim.x;
            if (t_col >= width)
                t_col = width - 1;
            s_input[(threadIdx.y + blockDim.y ) * shared_width + threadIdx.x + blockDim.x] = input[ d * width * height + t_row * width + t_col];
        }
    }

        __syncthreads();

        if(id_x < width && id_y < height){

        for(int ch = 0 ; ch < filterdepth ; ch ++){ // iter through number of filter
            //
            int out_idx = ch * width * height + id_y * width +id_x;
            double sum = 0;
                for(int x = 0 ; x < filterWidth ; x ++){
                    int row = threadIdx.y + x;
                    for(int y = 0 ; y < filterWidth ; y++){
                        int col = threadIdx.x + y;                        
                        sum += filter[ch*filterWidth*filterWidth*depth + d*filterWidth*filterWidth + x*filterWidth + y] * 
                            s_input[row*shared_width + col];
                    }
                }
            if(d == 0)
                output[out_idx] = sum + bias[ch];
            else
                output[out_idx] += sum;
        }
        }

     __syncthreads();

    }
}


#ifdef UnifiedMem

void CNN_cuda_v2::forward(const Matrix& _input){
    // in: _input
    // out: this->_foward;
    // filter.size.x : out_chanels;
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));

    _foward.setSize(_input.size.x, filter.size.x * out_cols *out_rows);

    dim3 blockSize(32,32);
    dim3 gridSize((in_cols-1)/blockSize.x+1,(in_rows-1)/blockSize.y+1);
    int shared_width = blockSize.x + filter_width - 1;
    int shared_height = blockSize.y + filter_width - 1;
    int shared_size = shared_width *  shared_height * in_channels;

    if (shared_size * sizeof(double) <= devProv.sharedMemPerBlock){ // enough shared mem/block to calculate all chanels in 1 kernel   
        if (padding)
            for(int sample = 0 ; sample < _foward.size.x ; sample ++){
                convPad<<<gridSize,blockSize,shared_size*sizeof(double)>>>(_input.data[sample],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample],shared_width, shared_height);
            }
        else
            for(int sample = 0 ; sample < _foward.size.x ; sample ++){
                convNonePad<<<gridSize,blockSize,shared_size*sizeof(double)>>>(_input.data[sample],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample],shared_width, shared_height);
            }
    }
    else{
        shared_size /= in_channels;
        if (padding)
            for(int sample = 0 ; sample < _foward.size.x ; sample ++){
                convPad1_shared<<<gridSize,blockSize,shared_size*sizeof(double)>>>(_input.data[sample],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample],shared_width, shared_height);
            }
        else
            for(int sample = 0 ; sample < _foward.size.x ; sample ++){
                convNonePad1_shared<<<gridSize,blockSize,shared_size*sizeof(double)>>>(_input.data[sample],in_cols,in_rows,in_channels,filter.data[0],filter_width,filter.size.x,bias.data[0],_foward.data[sample],shared_width, shared_height);
            }

    }
    cudaDeviceSynchronize();
}

#else

void CNN_cuda_v2::forward(const Matrix& _input){
    // in: _input
    // out: this->_foward;
    // filter.size.x : out_chanels;
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));

    _foward.setSize(_input.size.x, filter.size.x * out_cols *out_rows);

    dim3 blockSize(32,32);
    dim3 gridSize((in_cols-1)/blockSize.x+1,(in_rows-1)/blockSize.y+1);
    int shared_width = blockSize.x + filter_width - 1;
    int shared_height = blockSize.y + filter_width - 1;
    int shared_size = shared_width *  shared_height * in_channels;

    if (shared_size * sizeof(double) <= devProv.sharedMemPerBlock){ // enough shared mem/block to calculate all chanels in 1 kernel
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
                convPad<<<gridSize,blockSize,shared_size*sizeof(double)>>>(input,in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output,shared_width, shared_height);
                CHECK(cudaMemcpy(_foward.data[sample],output,outbytes,cudaMemcpyDeviceToHost));
            }
        else
            for(int sample = 0 ; sample < _foward.size.x ; sample ++){
                CHECK(cudaMemcpy(input,_input.data[sample],inbytes,cudaMemcpyHostToDevice));
                convNonePad<<<gridSize,blockSize,shared_size*sizeof(double)>>>(input,in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output,shared_width,shared_height);
                CHECK(cudaMemcpy(_foward.data[sample],output,outbytes,cudaMemcpyDeviceToHost));
            }
        cudaFree(input);
        cudaFree(output);
        cudaFree(d_bias);
        cudaFree(fil);
    }
    else{
        shared_size /= in_channels;
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
                convPad1_shared<<<gridSize,blockSize,shared_size*sizeof(double)>>>(input,in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output,shared_width, shared_height);
                CHECK(cudaMemcpy(_foward.data[sample],output,outbytes,cudaMemcpyDeviceToHost));
            }
        else
            for(int sample = 0 ; sample < _foward.size.x ; sample ++){
                CHECK(cudaMemcpy(input,_input.data[sample],inbytes,cudaMemcpyHostToDevice));
                convNonePad1_shared<<<gridSize,blockSize,shared_size*sizeof(double)>>>(input,in_cols,in_rows,in_channels,fil,filter_width,filter.size.x,d_bias,output,shared_width,shared_height);
                CHECK(cudaMemcpy(_foward.data[sample],output,outbytes,cudaMemcpyDeviceToHost));
            }
        cudaFree(input);
        cudaFree(output);
        cudaFree(d_bias);
        cudaFree(fil);
    
    }

}


#endif