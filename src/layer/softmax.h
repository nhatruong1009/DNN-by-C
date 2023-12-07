#ifndef SRC_LAYER_SOFTMAX_H_
#define SRC_LAYER_SOFTMAX_H_

#include "../layer.h"

class Softmax: public Layer {
 public:
  Softmax(){}
  ~Softmax(){}
  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
  void update(Optimizer& opt){}
};

#endif  // SRC_LAYER_SOFTMAX_H_
