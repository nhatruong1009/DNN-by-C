#ifndef __MATRIX_MATH__H__
#define __MATRIX_MATH__H__
#include <random>
#include "matrix.h"
static std::default_random_engine generator;


inline void set_normal_random_matrix(Matrix& _in, float mu, float sigma) {
  std::normal_distribution<float> distribution(mu, sigma);
  for (int i = 0; i < _in.size.x; i ++) {
    for (int k = 0 ; k < _in.size.y; k ++)
        _in.data[i][k] = distribution(generator);
  }
}


inline bool Col_toMatrix(const Matrix &_in,int col,int x, int y,Matrix &_out){
  if (_out.data == _in.data)
    return false;
  _out.setSize(x,y);
  for(int i = 0 ; i < x ; i++){
    for(int j = 0; j < y ; j ++){
        _out[i][j] = _in.data[col][i*y+j];
    }
  }
  return true;
}


inline int maxIdx(double *arr,int n){
  int idx =0;
  double max = arr[0];
  for(int i = 1 ; i < n ; i++){
    if (max < arr[i]){
      idx = i;
      max = arr[i];
    }
  }
  return idx;
}

inline double compute_accuracy(Matrix& _pred, Matrix& labels){
  int n_sample = _pred.size.x;
  int out = _pred.size.y;
  double points = 0;
  for(int i = 0 ; i <n_sample ; i++){
    points+= labels[i][maxIdx(_pred[i],out)]== 1;
  }
  return points/n_sample;
}



namespace numpy
{ 
    enum axis{
      x,y
    };

  
  void dot(const Matrix &A, const Matrix &B, Matrix &C);
  void Transpose(const Matrix &_in, Matrix &_out);
  
  void add(const Matrix &in1,const Matrix &in2,Matrix &out);
  void sub(const Matrix &in1,const Matrix &in2,Matrix &out);
  void mul(const Matrix &in1,const Matrix &in2,Matrix &out);
  void div(const Matrix &in1,const Matrix &in2,Matrix &out);
  void add(const Matrix &in,double val,Matrix &out);
  void sub(const Matrix &in,double val,Matrix &out);
  void mul(const Matrix &in,double val,Matrix &out);
  void div(const Matrix &in,double val,Matrix &out);
  void add(double val,const Matrix &in,Matrix &out);
  void sub(double val,const Matrix &in,Matrix &out);
  void mul(double val,const Matrix &in,Matrix &out);
  void div(double val,const Matrix &in,Matrix &out);
  void reduce_sum(const Matrix &_in,Matrix&_out, axis _reduce);
  void reduce_max(const Matrix &_in,Matrix&_out, axis _reduce);
  void reduce_min(const Matrix &_in,Matrix&_out, axis _reduce);
  double sum(const Matrix &_in);
} // namespace np


#endif