#ifndef __AVG_POOLING__
#define __AVG_POOLING__
#include "../layer.h"


class Avg_pooling : public Layer{
private:
    int kernel_size;
    int in_channels;
    int in_rows;
    int in_cols;
    int stride;
    
public:
    Avg_pooling(int in_chanels,int in_rows, int in_cols,int kernel_size,int stride) : 
        in_channels(in_chanels),in_rows(in_rows), in_cols(in_cols), kernel_size(kernel_size), stride(stride){};
    void forward(const Matrix& mal);
    ~Avg_pooling(){};
    void backward(const Matrix& _in, const Matrix& _out);
    void update(Optimizer& opt){}
};

#endif