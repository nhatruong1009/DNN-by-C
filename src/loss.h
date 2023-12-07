#ifndef _ERROR_OUT_
#define _ERROR_OUT_

#include "matrix.h"
class Loss {
 protected:
  float loss;  // value of loss

 public:
  virtual ~Loss() {}
   Matrix grad_bottom;  // gradient w.r.t input

  virtual void evaluate(const Matrix& pred, const Matrix& target) = 0;
  virtual float output() { return loss; }
  virtual const Matrix& back_gradient() { return grad_bottom; }
};


class MSE: public Loss {
public:
  void evaluate(const Matrix& pred, const Matrix& target);
};


class Cross_entropy:public Loss{
public:
    void evaluate(const Matrix& pred, const Matrix& target);
};

#endif