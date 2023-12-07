#include "ReLU.h"
#include "../matrix_math.h"

double relu(double a){
    return a>0 ? a : 0;
}

void ReLU::forward(const Matrix& mal){
    this->_foward = mal;
    this->_foward.apply(relu);
}

double is_pos(double a){
    return a>0.0;
}

void ReLU::backward(const Matrix& _in, const Matrix& _out){
    Matrix positive;
    positive = _in;
    positive.apply(is_pos);
    numpy::mul(_out,positive,_backward);
    //_backward.printSize();
}