#include "softmax.h"
#include "../matrix_math.h"

void Softmax::forward(const Matrix& bottom){
    _foward = bottom;
    _foward.apply(exp);
    Matrix sum; 
    numpy::reduce_sum(_foward,sum,numpy::axis::y);
    _foward/=sum;
}
void Softmax::backward(const Matrix& bottom, const Matrix& grad_top){
    Matrix temp;
    numpy::mul(_foward,grad_top,temp);
    numpy::reduce_sum(temp,temp,numpy::axis::y);
    numpy::sub(grad_top,temp,temp);
    numpy::mul(_foward,temp,_backward);
}