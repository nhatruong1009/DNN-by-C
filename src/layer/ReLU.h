#ifndef __RELU_LAYER__
#define __RELU_LAYER__
#include "../layer.h"

class ReLU : public Layer{
    public:
    ReLU(){}
    ~ReLU(){};
    void forward(const Matrix&);
    void backward(const Matrix&,const Matrix&);
    void update(Optimizer& opt){}
};

#endif