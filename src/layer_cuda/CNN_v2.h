#include "../layer/convolution.h"


class CNN_cuda_v2: public Convolution{

/*
upgrade from version 1, using shared memory
*/

public:
  CNN_cuda_v2(int in_channels,int in_rows, int in_cols,int out_channels,int filter_width,bool padding):Convolution(in_channels,in_rows,in_cols,out_channels,filter_width,padding){}
  void forward(const Matrix& _input);  
};