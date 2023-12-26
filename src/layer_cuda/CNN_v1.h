#include "../layer/convolution.h"


class CNN_cuda_v1: public Convolution{

/*
standard version, not using any special technique
*/

public:
  CNN_cuda_v1(int in_channels,int in_rows, int in_cols,int out_channels,int filter_width,bool padding):Convolution(in_channels,in_rows,in_cols,out_channels,filter_width,padding){}
  void forward(const Matrix& _input);  
};