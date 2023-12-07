#include "loss.h"

#include "matrix_math.h"



void MSE::evaluate(const Matrix& pred, const Matrix& target) {
  int n = pred.size.x;
  // forward: L = sum{ (p-y).*(p-y) } / n
  numpy::sub(pred,target,grad_bottom);
  Matrix temp;
  numpy::mul(grad_bottom,grad_bottom,temp);
  loss = numpy::sum(temp);
  loss /= n;
  //Matrix diff = pred - target;
  //loss = diff.cwiseProduct(diff).sum();
  //loss /= n;
  // backward: d(L)/d(p) = (p-y)*2/n
  //grad_bottom = diff * 2 / n;
  grad_bottom /= (0.5f * n);
}

void Cross_entropy::evaluate(const Matrix& pred, const Matrix& target){
  int n = pred.size.x;
  const double eps = 1e-8;
  Matrix _pred_eps;
  numpy::add(pred,eps,_pred_eps);
  Matrix temp;
  temp = _pred_eps;
  temp.apply(log);
  numpy::mul(target,temp,temp);
  loss = -numpy::sum(temp);
  loss /=n;
  numpy::div(target,_pred_eps,grad_bottom);
  grad_bottom/=n;
  grad_bottom*=-1;

}