#ifndef _LAYER_MODE__
#define _LAYER_MODE__

#include "matrix.h"
#include "optimizer.h"
class Layer {
public:
  Matrix _foward;  // layer output
  Matrix _backward;  // gradient w.r.t input

  
  Matrix grad_weight;  // gradient w.r.t weight
  Matrix grad_bias;;
public:
  virtual ~Layer(){};
  virtual void forward(const Matrix&) = 0;
  virtual void backward(const Matrix&,const Matrix&) = 0;
  virtual void update(Optimizer& opt) =0 ;
};


#endif