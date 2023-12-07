#include "matrix_math.h"
#include <string>

void numpy::dot(const Matrix &A, const Matrix &B, Matrix &C) {
    if (C.data == A.data || C.data == B.data){
        Matrix temp;
        numpy::dot(A,B,temp);
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

void numpy::Transpose(const Matrix &_in, Matrix &_out){
    if(_out.data == _in.data){
        Matrix temp;
        numpy::Transpose(_in,temp);
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


void numpy::add(const Matrix &in1,const Matrix &in2,Matrix &out){
    if (out.data == in1.data || out.data == in2.data){
        Matrix temp;
        numpy::add(in1,in2,temp);
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
void numpy::sub(const Matrix &in1,const Matrix &in2,Matrix &out){
    if (out.data == in1.data || out.data == in2.data){
        Matrix temp;
        numpy::sub(in1,in2,temp);
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
void numpy::mul(const Matrix &in1,const Matrix &in2,Matrix &out){
    if (out.data == in1.data || out.data == in2.data){
        Matrix temp;
        numpy::mul(in1,in2,temp);
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
void numpy::div(const Matrix &in1,const Matrix &in2,Matrix &out){
    if (out.data == in1.data || out.data == in2.data){
        Matrix temp;
        numpy::div(in1,in2,temp);
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
void numpy::add(const Matrix &in,double val,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::add(in,val,temp);
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
void numpy::sub(const Matrix &in,double val,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::sub(in,val,temp);
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
void numpy::mul(const Matrix &in,double val,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::mul(in,val,temp);
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
void numpy::div(const Matrix &in,double val,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::div(in,val,temp);
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



void numpy::add(double val,const Matrix &in,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::add(val,in,temp);
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
void numpy::sub(double val,const Matrix &in,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::sub(val,in,temp);
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
void numpy::mul(double val,const Matrix &in,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::mul(val,in,temp);
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
void numpy::div(double val,const Matrix &in,Matrix &out){
    if (out.data == in.data){
        Matrix temp;
        numpy::div(val,in,temp);
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

void numpy::reduce_sum(const Matrix &_in,Matrix&_out, axis _reduce){

    if(_out.data == _in.data){
        Matrix temp;
        numpy::reduce_sum(_in,temp,_reduce);
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

void numpy::reduce_max(const Matrix &_in,Matrix&_out, axis _reduce){
    if(_out.data == _in.data){
        Matrix temp;
        numpy::reduce_max(_in,temp,_reduce);
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


void numpy::reduce_min(const Matrix &_in,Matrix&_out, axis _reduce){
    if(_out.data == _in.data){
        Matrix temp;
        numpy::reduce_min(_in,temp,_reduce);
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



double numpy::sum(const Matrix &_in){
      double temp = 0;
      for(int i = 0 ; i < _in.size.x; i++){
        for(int k = 0 ; k < _in.size.y; k++)
          temp+= _in.data[i][k];
      }
      return temp;
    }
