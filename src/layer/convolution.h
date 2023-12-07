#ifndef __CONVOLUTION__
#define __CONVOLUTION__
#include "../layer.h"


class Convolution : public Layer{
public:
    int filter_width;
    Matrix filter;
    Matrix bias;
    bool padding;
    int in_channels;
    int in_rows;
    int in_cols;
    int out_rows;
    int out_cols;
public:
    void col2im(const Matrix& data_col, double* image);
    void convSample(const double* image, double*& output);
    Convolution(int in_channels,int in_rows, int in_cols,int out_chanels,int filer_width,bool padding = false);
    void forward(const Matrix& _input);
    void backward(const Matrix& _in, const Matrix& _out);
    void update(Optimizer& opt);
    ~Convolution(){};
    void getpara(){
        printf("--- conv ---\n");
        filter.printSize();
        filter.print();
        bias.printSize();
        bias.print();
    }
};

#endif