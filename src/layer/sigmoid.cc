#include "sigmoid.h"
#include <math.h>
#include "../matrix_math.h"
double sigmoid(double a){
    return 1.0/(1.0+exp(-a));
}

void Sigmoid::forward(const Matrix& mal){
    this->_foward = mal;
    this->_foward.apply(sigmoid);
}

double dir_sigmoid(double a){
    return a*(1-a);
}

void Sigmoid::backward(const Matrix& _in, const Matrix& grad_in){

    Matrix da_dz;
    numpy::sub(1.0,this->_foward,da_dz); // 1-out
    numpy::mul(this->_foward,da_dz,da_dz);// out(1-out)
    numpy::mul(grad_in,da_dz,_backward); 
    //_backward.printSize();
}