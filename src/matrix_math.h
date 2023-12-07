#ifndef __MATRIX_MATH__H__
#define __MATRIX_MATH__H__
#include "matrix.h"
#include <random>

static std::default_random_engine generator;

inline void set_normal_random_matrix(Matrix& matrix, float mu, float sigma) {
  std::normal_distribution<float> distribution(mu, sigma);
  for (int i = 0; i < matrix.size.x; i ++) {
    for (int k = 0 ; k < matrix.size.y; k ++)
        matrix.data[i][k] = distribution(generator);
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
    void dot(const Matrix &_right, const Matrix &_left, Matrix &_out);
    void Transpose(const Matrix &_in,Matrix &_out);
    void add(const Matrix &_in1,const Matrix &_in2,Matrix &_out);
    void sub(const Matrix &_in1,const Matrix &_in2,Matrix &_out);
    void mul(const Matrix &_in1,const Matrix &_in2,Matrix &_out);
    void div(const Matrix &_in1,const Matrix &_in2,Matrix &_out);

    void add(const Matrix &_in,double _val,Matrix &_out);
    void sub(const Matrix &_in,double _val,Matrix &_out);
    void mul(const Matrix &_in,double _val,Matrix &_out);
    void div(const Matrix &_in,double _val,Matrix &_out);

    void add(double _val,const Matrix &_in,Matrix &_out);
    void sub(double _val,const Matrix &_in,Matrix &_out);
    void mul(double _val,const Matrix &_in,Matrix &_out);
    void div(double _val,const Matrix &_in,Matrix &_out);

    enum axis{
      x,y
    };

    void reduce_sum(const Matrix &_in, Matrix&_out, axis _reduce = axis::x);
    double sum(const Matrix &_in);

    void reduce_max(const Matrix &_in, Matrix&_out, axis _reduce = axis::x);
    void reduce_min(const Matrix &_in, Matrix&_out, axis _reduce = axis::x);
} // namespace np


#endif