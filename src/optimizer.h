#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "matrix.h"

class Optimizer {
 protected:
  float lr;  // learning rate
  float decay;  // weight decay factor (default: 0)

 public:
  explicit Optimizer(float lr = 0.01, float decay = 0.0) :
                     lr(lr), decay(decay) {}
  virtual ~Optimizer() {}

  virtual void update(Matrix& w,
                      Matrix& dw) = 0;
};

#endif  // SRC_OPTIMIZER_H_
