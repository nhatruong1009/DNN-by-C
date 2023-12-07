#include "avgPooling.h"


void Avg_pooling::forward(const Matrix& mal){ // input z: number of sample, y: number of col eachs sample, x: number of row * depth each sample 

    int out_rows = (in_rows-kernel_size)/stride + 1;
    int out_cols = (in_cols-kernel_size)/stride + 1;

    _foward.setSize(mal.size.x,out_rows*out_cols*in_channels);
    
    for (int sample = 0 ; sample < mal.size.x ; sample ++){

        for(int i = 0 ; i < out_rows ; i ++){
            int beg_r = i*stride;
            for(int j = 0 ; j < out_cols ; j ++){
                int beg_c = j*stride;
                double avg = 0;

                for(int k_r = 0 ; k_r < kernel_size; k_r ++){
                    for(int k_c = 0 ; k_c < kernel_size; k_c ++){
                        avg += mal.data[sample][(beg_r+k_r)*in_cols+beg_c+k_c];
                    }

                }
                _foward[sample][i*out_cols+j] = avg/(kernel_size*kernel_size);
            }

        }

    }
}


void Avg_pooling::backward(const Matrix& _in, const Matrix& _out){
    _backward.setSize(_in.size);
    _backward.setval(0);

    int out_rows = (in_rows-kernel_size)/stride + 1;
    int out_cols = (in_cols-kernel_size)/stride + 1;

    //_backward.printSize();
    int _back_size = in_rows * in_cols;
    int _out_size = out_rows*out_cols;

    for(int i = 0 ; i < _in.size.x; i ++){
        for(int c = 0 ; c < in_channels;  c++){
            for(int r_out = 0 ; r_out < out_rows; r_out++){
                int beg_r = r_out*stride;
                for(int c_out = 0; c_out < out_cols ; c_out++){
                    int beg_c = c_out*stride;

                    for(int r_pool = 0 ; r_pool < kernel_size; r_pool++){
                        int x = beg_r + r_pool;
                        for(int c_pool =0; c_pool < kernel_size; c_pool ++){
                            int y = beg_c + c_pool;
                            if(
                                x >= in_rows || y >=in_cols
                            )
                            continue;
                            _backward[i][c* _back_size + x*in_cols +y] += _out.data[i][c * _out_size + r_out * out_cols + c_out];
                        }
                    }

                }
            }
        }
    }

}