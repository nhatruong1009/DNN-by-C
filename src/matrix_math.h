#ifndef __MATRIX_MATH__H__
#define __MATRIX_MATH__H__
#include <random>
#include "matrix.h"
static std::default_random_engine generator;

template<class T>
inline void set_normal_random_matrix(T& _in, float mu, float sigma) {
  std::normal_distribution<float> distribution(mu, sigma);
  for (int i = 0; i < _in.size.x; i ++) {
    for (int k = 0 ; k < _in.size.y; k ++)
        _in.data[i][k] = distribution(generator);
  }
}

template<class T>
inline bool Col_toMatrix(const T &_in,int col,int x, int y,T &_out){
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
template<class T>
inline double compute_accuracy(T& _pred, T& labels){
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

  template<class T>
  void dot(const T &A, const T &B, T &C) {
      if (C.data == A.data || C.data == B.data){
          T temp;
          dot(A,B,temp);
          swap(temp,C);
          return;
      }
      C.setSize(A.size.x,B.size.y);
      for (int r = 0 ; r < A.size.x ; r ++){
          for (int c = 0 ; c < B.size.y ; c ++){
            double v = 0;
            for (int k = 0 ; k < A.size.y ; k ++){
              v+= A.data[r][k] * B.data[k][c];
            }
            C[r][c] = v;
          }
        }
      return;
  }

  template<class T>
  void Transpose(const T &_in, T &_out){
      if(_out.data == _in.data){
          T temp;
          Transpose(_in,temp);
          swap(_out,temp);
          return;
      }

      _out.setSize(_in.size.y,_in.size.x);
      for(int r = 0 ; r < _out.size.x; r++){
          for(int c = 0 ; c < _out.size.y; c++){
              _out[r][c] = _in.data[c][r];
          }
      }
      return;
  }

  template<class T>
  void add(const T &in1,const T &in2,T &out){
      if (out.data == in1.data || out.data == in2.data){
          T temp;
          add(in1,in2,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in1.size);
      if (in2.size.x == 0 || in2.size.y == 0)
          return ;
      if( in1.size.x == in2.size.x){
          bool is_col = !(in2.size.y == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k] +in2.data[i][k*is_col];
              }
          }
          return;
      }
      else if (in1.size.y == in2.size.y){
          bool is_row = !(in2.size.x == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k] + in2.data[i*is_row][k];
              }
          }
          return;
      }
      return;
  };

  template<class T>
  void sub(const T &in1,const T &in2,T &out){
      if (out.data == in1.data || out.data == in2.data){
          T temp;
          sub(in1,in2,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in1.size);
      if (in2.size.x == 0 || in2.size.y == 0)
          return ;
      if( in1.size.x == in2.size.x){
          bool is_col = !(in2.size.y == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k]  - in2.data[i][k*is_col];
              }
          }
          return;
      }
      else if (in1.size.y == in2.size.y){
          bool is_row = !(in2.size.x == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k] - in2.data[i*is_row][k];
              }
          }
          return;
      }
      return;
  };

  template<class T>
  void mul(const T &in1,const T &in2,T &out){
      if (out.data == in1.data || out.data == in2.data){
          T temp;
          mul(in1,in2,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in1.size);
      if (in2.size.x == 0 || in2.size.y == 0)
          return ;
      if( in1.size.x == in2.size.x){
          bool is_col = !(in2.size.y == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k]  * in2.data[i][k*is_col];
              }
          }
          return;
      }
      else if (in1.size.y == in2.size.y){
          bool is_row = !(in2.size.x == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k] * in2.data[i*is_row][k];
              }
          }
          return;
      }
      return;
  };

  template<class T>
  void div(const T &in1,const T &in2,T &out){
      if (out.data == in1.data || out.data == in2.data){
          T temp;
          div(in1,in2,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in1.size);
      if (in2.size.x == 0 || in2.size.y == 0)
          return ;
      if( in1.size.x == in2.size.x){
          bool is_col = !(in2.size.y == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k] / in2.data[i][k*is_col];
              }
          }
          return;
      }
      else if (in1.size.y == in2.size.y){
          bool is_row = !(in2.size.x == 1);
          for(int i = 0 ; i < in1.size.x; i++){
              for(int k = 0 ; k < in1.size.y ; k ++){
                  out.data[i][k] = in1.data[i][k] / in2.data[i*is_row][k];
              }
          }
          return;
      }
      return;
  };

  template<class T>
  void add(const T &in,double val,T &out){
      if (out.data == in.data){
          T temp;
          add(in,val,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = in.data[i][k] + val;
          }
      }
      return;
  };

  template<class T>
  void sub(const T &in,double val,T &out){
      if (out.data == in.data){
          T temp;
          sub(in,val,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = in.data[i][k] - val;
          }
      }
      return;
  };

  template<class T>
  void mul(const T &in,double val,T &out){
      if (out.data == in.data){
          T temp;
          mul(in,val,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = in.data[i][k] * val;
          }
      }
      return;
  };

  template<class T>
  void div(const T &in,double val,T &out){
      if (out.data == in.data){
          T temp;
          div(in,val,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = in.data[i][k] / val;
          }
      }
      return;
  };


  template<class T>
  void add(double val,const T &in,T &out){
      if (out.data == in.data){
          T temp;
          add(val,in,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = val + in.data[i][k];
          }
      }
      return;
  };

  template<class T>
  void sub(double val,const T &in,T &out){
      if (out.data == in.data){
          T temp;
          sub(val,in,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = val - in.data[i][k];
          }
      }
      return;
  };

  template<class T>
  void mul(double val,const T &in,T &out){
      if (out.data == in.data){
          T temp;
          mul(val,in,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = val * in.data[i][k] ;
          }
      }
      return;
  };

  template<class T>
  void div(double val,const T &in,T &out){
      if (out.data == in.data){
          T temp;
          div(val,in,temp);
          swap(out,temp);
          return;
      }
      out.setSize(in.size);
      for(int i = 0 ; i < in.size.x ; i ++){
          for(int k = 0 ; k < in.size.y;k++){
              out.data[i][k] = val / in.data[i][k] ;
          }
      }
      return;
  };

  template<class T>
  void reduce_sum(const T &_in,T&_out, axis _reduce){

      if(_out.data == _in.data){
          T temp;
          reduce_sum(_in,temp,_reduce);
          swap(_out,temp);
          return;
      }
      if (_reduce == axis::x){
          _out.setSize(1,_in.size.y);
          _out.setval(0);
          for(int i = 0 ; i < _in.size.y ; i++){
              for(int j = 0 ; j < _in.size.x ; j++){
                  _out[0][i] += _in.data[j][i];
              }
          }
      }
      else{
          _out.setSize(_in.size.x,1);
          _out.setval(0);
          for(int i = 0 ; i < _in.size.y ; i++){
              for(int j = 0 ; j < _in.size.x ; j++){
                  _out[j][0] += _in.data[j][i];
              }
          }
      }
  }

  template<class T>
  void reduce_max(const T &_in,T&_out, axis _reduce){
      if(_out.data == _in.data){
          T temp;
          reduce_max(_in,temp,_reduce);
          swap(_out,temp);
          return;
      }
      if (_reduce == axis::x){
          _out.setSize(1,_in.size.y);
          for(int i=0 ; i < _in.size.y ; i++)
              _out[0][i] = _in.data[0][i];
          for(int l = 0 ; l < _in.size.y ; l++){
              for(int i = 0 ; i< _in.size.x ; i++){
                  if (_out[0][l] < _in.data[i][l])
                      _out[0][l] = _in.data[i][l];
              }
          }

      }
      else{
          _out.setSize(_in.size.x,1);
          _out.setval(0);
          for(int i=0 ; i < _in.size.x ; i++)
              _out[i][0] = _in.data[i][0];
          for(int l = 0 ; l < _in.size.x ; l++){
              for(int i = 0 ; i< _in.size.y ; i++){
                  if (_out[l][0] < _in.data[l][i])
                      _out[l][0] = _in.data[l][i];
              }
          }

      }
  }

  template<class T>
  void reduce_min(const T &_in,T&_out, axis _reduce){
      if(_out.data == _in.data){
          T temp;
          reduce_min(_in,temp,_reduce);
          swap(_out,temp);
          return;
      }
      if (_reduce == axis::x){
          _out.setSize(1,_in.size.y);
          for(int i=0 ; i < _in.size.y ; i++)
              _out[0][i] = _in.data[0][i];
          for(int l = 0 ; l < _in.size.y ; l++){
              for(int i = 0 ; i< _in.size.x ; i++){
                  if (_out[0][l] > _in.data[i][l])
                      _out[0][l] = _in.data[i][l];
              }
          }

      }
      else{
          _out.setSize(_in.size.x,1);
          _out.setval(0);
          for(int i=0 ; i < _in.size.x ; i++)
              _out[i][0] = _in.data[i][0];
          for(int l = 0 ; l < _in.size.x ; l++){
              for(int i = 0 ; i< _in.size.y ; i++){
                  if (_out[l][0] > _in.data[l][i])
                      _out[l][0] = _in.data[l][i];
              }
          }

      }
  }


  template<class T>
  double sum(const T &_in){
        double temp = 0;
        for(int i = 0 ; i < _in.size.x; i++){
          for(int k = 0 ; k < _in.size.y; k++)
            temp+= _in.data[i][k];
        }
        return temp;
      }


} // namespace np


#endif