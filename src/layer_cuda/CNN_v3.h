#include "../layer/convolution.h"


class CNN_cuda_v3: public Convolution{

/*
using technique that transform convolution into matrix-matrix multiplication.
*/
private:
  void TransformFillter(double* filter_out,int n, int in_height,int in_width);

public:
  CNN_cuda_v3(int in_channels,int in_rows, int in_cols,int out_channels,int filter_width,bool padding):Convolution(in_channels,in_rows,in_cols,out_channels,filter_width,padding){}
  void forward(const Matrix& _input);  
};