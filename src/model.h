#ifndef __DNN_MODEL__
#define __DNN_MODEL__
#include "layer.h"
#include "matrix.h"
#include <vector>
#include "loss.h"

class Model : public Layer{
    public:
    std::vector<Layer*> layers;
    Loss* loss_layer;
    public:
    Model(){}
    ~Model(){}
    void add(Layer & _Layers);
    void forward(const Matrix& _input);
    void addloss(Loss & loss){
        loss_layer = & loss;
    };
    void info(){
        for(int i = 0; i< layers.size(); i ++){
            layers[i]->_foward.printSize();
        }
    }
    void backward(const Matrix& _input,const Matrix& _target);
    void update(Optimizer& opt);
};

#endif