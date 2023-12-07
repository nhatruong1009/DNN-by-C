#ifndef __MAX_POOLING__
#define __MAX_POOLING__
#include "../layer.h"


class Max_pooling : public Layer{
private:
    int kernel_size;
    int in_channels;
    int in_rows;
    int in_cols;
    int stride;
    Matrix max_idxs;

public:
    Max_pooling(int in_chanels,int in_rows, int in_cols,int kernel_size,int stride) : 
        in_channels(in_chanels),in_rows(in_rows), in_cols(in_cols), kernel_size(kernel_size), stride(stride){};
    void forward(const Matrix& mal);
    ~Max_pooling(){};
    void backward(const Matrix& _in, const Matrix& _out);
    void update(Optimizer& opt){}
};

#endif