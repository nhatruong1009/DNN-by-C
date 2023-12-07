#include "maxPooling.h"

void Max_pooling::forward(const Matrix& mal){ // input z: number of sample, y: number of col eachs sample, x: number of row * depth each sample 

    int out_rows = (in_rows-kernel_size)/stride + 1;
    int out_cols = (in_cols-kernel_size)/stride + 1;

    _foward.setSize(mal.size.x,out_rows*out_cols*in_channels);
    max_idxs.setSize(_foward.size);
    for (int sample = 0 ; sample < mal.size.x ; sample ++){

        for(int i = 0 ; i < out_rows ; i ++){
            int beg_r = i*stride;
            for(int j = 0 ; j < out_cols ; j ++){
                int beg_c = j*stride;
                double max = mal.data[sample][beg_r*in_cols+beg_c];
                max_idxs[sample][i*in_cols+j] = beg_r*in_cols+beg_c;

                for(int k_r = 0 ; k_r < kernel_size; k_r ++){
                    for(int k_c = 0 ; k_c < kernel_size; k_c ++){
                        if(max < mal.data[sample][(beg_r+k_r)*in_cols+beg_c+k_c]){
                            max = mal.data[sample][(beg_r+k_r)*in_cols+beg_c+k_c];
                            max_idxs[sample][i*in_cols+j] = (beg_r+k_r)*in_cols+beg_c+k_c;
                        }

                    }

                }
                _foward[sample][i*out_cols+j] = max;
            }

        }

    }
}


void Max_pooling::backward(const Matrix& _in, const Matrix& _out){
    _backward.setSize(_in.size);
    _backward.setval(0);
    for(int i = 0 ; i < max_idxs.size.x; i++){
        for(int j = 0 ; j < max_idxs.size.y; j++){
            _backward[i][(int)max_idxs[i][j]] += _out.data[i][j];
        }
    }
    //_backward.printSize();

}