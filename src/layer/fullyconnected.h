#ifndef __FULLY_LAYER__
#define __FULLY_LAYER__

#include "../layer.h"
#include "../matrix.h"

class FullyConnected : public Layer {
    private:
    const int dim_in;
    const int dim_out;
    void init();

    public:
    Matrix weight;
    Matrix bias;

    FullyConnected(int dim_in,int dim_out);
    void forward(const Matrix& mal);
    ~FullyConnected();
    void backward(const Matrix&,const Matrix&);
    void update(Optimizer& opt);


    void check(){
        this->weight.print();
        printf("\n");
        this->bias.print();
        printf("\n");
        this->_foward.print();
    }
};

#endif