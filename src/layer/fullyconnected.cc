#include "fullyconnected.h"
#include <random>
#include "../matrix_math.h"

#define PRINT_DEBUG


void FullyConnected::init(){
    this->weight.setSize(this->dim_in,this->dim_out);
    this->bias.setSize(1,this->dim_out);
    set_normal_random_matrix(this->weight,0,0.01);
    set_normal_random_matrix(this->bias,0,0.01);
}

FullyConnected::FullyConnected(const int _dim_in, const int _dim_out) : dim_in(_dim_in), dim_out(_dim_out){
    init();
}

void FullyConnected::forward(const Matrix& mal){
    numpy::dot(mal,this->weight,this->_foward);
    this->_foward += this->bias;
}

void FullyConnected::backward(const Matrix& _out,const Matrix& _grad_in){
    const int n_sample = _out.size.x;
    // -out sample x in dim
    // -gradin: sample x outdim
    Matrix temp;
    numpy::Transpose(_grad_in,temp);
    numpy::dot(temp,_out,grad_weight);
    numpy::reduce_sum(_grad_in,grad_bias,numpy::axis::x);
    numpy::Transpose(weight,temp);
    numpy::dot(_grad_in,temp,_backward);
    //_backward.printSize();
    numpy::Transpose(grad_weight,grad_weight);
    //grad_weight = bottom * grad_top.transpose();
    //grad_bias = grad_top.rowwise().sum();
    // d(L)/d(x) = w * d(L)/d(z)
    //grad_bottom.resize(dim_in, n_sample);
    //grad_bottom = weight * grad_top;

}

void FullyConnected::update(Optimizer& opt){
    opt.update(weight,grad_weight);
    opt.update(bias,grad_bias);
}

FullyConnected::~FullyConnected(){}