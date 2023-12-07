#ifndef __SIGMOID_LAYER__
#define __SIGMOID_LAYER__
#include "../layer.h"

class Sigmoid : public Layer{
    public:
    Sigmoid(){}
    ~Sigmoid(){};
    void forward(const Matrix&);
    void backward(const Matrix&,const Matrix&);
    void update(Optimizer& opt){}
};

#endif