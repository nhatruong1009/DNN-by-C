#include "sgd.h"
#include "../matrix_math.h"
void SGD::update(Matrix& w, Matrix& dw) {
    Matrix& v = v_map[dw.data];
    if(v.size.x == 0){
        v.setSize(dw.size);
        v.setval(0);
    }
    Matrix temp;
    numpy::mul(decay,w,temp);
    numpy::add(dw,temp,temp);
    Matrix temp2;
    numpy::mul(momentum,v,temp2);
    numpy::add(temp2,temp,v);
    if (nesterov){
        numpy::mul(momentum,v,temp2);
        numpy::add(temp2,temp,temp2);
        numpy::mul(lr,temp2,temp2);
        w-=temp2;
    }
    else{
        numpy::mul(lr,v,temp2);
        w-=temp2;
    }
}
