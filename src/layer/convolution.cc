#include "convolution.h"
#include "../matrix_math.h"
#include <cstring>
Convolution::Convolution(int in_channels,int in_rows, int in_cols,int out_channels,int filter_width,bool padding){
    this->filter_width = filter_width;
    this->in_channels = in_channels;
    this->in_rows = in_rows;
    this->in_cols = in_cols;
    //filter.setSize(in_channels,filter_width*filter_width,out_channels); 
    filter.setSize(out_channels,in_channels*filter_width*filter_width);
    this->padding = padding;
    this->bias.setSize(1,out_channels);

    grad_weight = filter;
    grad_bias = bias;
    set_normal_random_matrix(this->filter,1.0f/(filter_width*filter_width*out_channels),0.01);
    set_normal_random_matrix(this->bias,1,0.01);

    out_rows = in_rows - (filter_width - 1)*(!padding);
    out_cols = in_cols - (filter_width - 1)*(!padding);
    
}

void Convolution::convSample(const double* image, double*& output){
   
    int pad = filter_width/2;
    
    for(int ch = 0 ; ch < filter.size.x ; ch ++){ // chanels out
        for(int i = 0 ; i < out_rows ; i ++){ // row idx out
            int beg_r = i - pad*padding;
            for(int j = 0 ; j < out_cols ; j ++){ // col idx out
                int beg_c = j - pad*padding;
                double kq = 0;

                for (int in_ch = 0; in_ch < in_channels ;in_ch ++){
                    for(int in_r = 0 ; in_r < filter_width ; in_r ++){
                        int x = beg_r + in_r;
                        if (x < 0)
                            x = 0;
                        else if(x >= in_rows)
                            x = in_rows-1;

                        for(int in_c = 0; in_c <filter_width ; in_c ++){
                            int y = beg_c + in_c;
                            if (y < 0)
                                y = 0;
                            else if(y >= in_cols)
                                y = in_cols-1;

                            kq += filter[ch][ in_ch*filter_width*filter_width + in_r* filter_width + in_c] * image[in_ch * in_rows * in_cols + x * in_cols + y];

                        }
                    }
                }

                output[ch*out_rows*out_cols + i * out_cols + j] = kq + bias[0][ch];
            }   
        }
    }
}

void Convolution::forward(const Matrix& _input){ 
    _foward.setSize(_input.size.x, filter.size.x * out_cols * out_rows);
    for(int sample = 0 ; sample < _foward.size.x ; sample ++){
        convSample(_input.data[sample],_foward.data[sample]);
    }

}

void Convolution::col2im(const Matrix& data_col, double* image) {
  int hw_in = in_rows * in_cols;
  int hw_kernel = filter_width * filter_width;
  int hw_out =out_rows * out_cols;
  // col2im
  for (int c = 0; c < in_channels; c ++) {
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / out_cols;
      int step_w = i % out_cols;
      int start_idx = step_h * in_cols * 1 + step_w * 1;  // left-top idx of window

      
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % in_cols + j % filter_width - (filter_width/2)*padding;  // col after padding
        int cur_row = start_idx / in_cols + j / filter_width - (filter_width/2)*padding;
        if (cur_col < 0 || cur_col >= in_cols || cur_row < 0 || cur_row >= in_rows) {
          continue;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * in_cols + cur_col;
          image[c * hw_in + pick_idx] += data_col.data[c * hw_kernel + j][i];  // pick which pixel
        }
      
      
      }
    }
  }
}


void Convolution::backward(const Matrix& _in, const Matrix& grad_in){
    int n_sample = _in.size.x;
    int channel_out = filter.size.x;
    grad_weight.setval(0);
    grad_bias.setval(0);
    _backward.setSize(n_sample,in_channels*in_rows*in_cols);
    _backward.setval(0);
    for(int i = 0 ; i < n_sample ; i++){
        Matrix grad_top_i;
        Col_toMatrix(grad_in,i,channel_out,out_rows*out_cols,grad_top_i);
        Matrix data_col;
        Col_toMatrix(_in,i,in_channels,in_rows*in_cols,data_col);
        numpy::Transpose(data_col,data_col);
        Matrix temp;
        numpy::dot(data_col,grad_top_i,temp);
        grad_weight+=temp;
        numpy::reduce_sum(grad_top_i,temp,numpy::axis::y);
        numpy::Transpose(temp,temp);
        grad_bias += temp;

        Matrix grad_bottom_i_col;
        numpy::Transpose(filter,temp);
        numpy::dot(temp,grad_top_i,grad_bottom_i_col);
        col2im(grad_bottom_i_col,_backward[i]);
    }
    //_backward.printSize();

}  

void Convolution::update(Optimizer& opt){
    opt.update(filter,grad_weight);
    opt.update(bias,grad_bias);
}