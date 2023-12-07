#include "matrix.h"
#include <cstring>


extern FILE * fil;

Matrix::Matrix(){
    this->size.x = 0;
    this->size.y = 0;
    this->data = NULL;
}

Matrix::Matrix(int x,int y,bool oneblock){   
    this->size.x = x;
    this->size.y = y;
    this->oneblock = oneblock;
    this->data = (double**)malloc(size.x*sizeof(double*));

    if (oneblock == true){
        data[0] = (double*)malloc(size.x*size.y*sizeof(double));
        if(data[0] != NULL){
            for(int i = 1 ; i < size.x ; i++){
                data[i] = data[0] + size.y*i; // this is legit :))
            }
            return;
        }
        this->oneblock = false;
    }
    for(int i = 0; i < size.x ; i++){
        data[i] = (double*)malloc(size.y*sizeof(double));
    }
}

Matrix::Matrix(const Matrix& mal){
    this->setSize(mal.size);
    for(int i = 0 ; i < size.x; i++)
        memcpy(data[i], mal.data[i], size.y *sizeof(double));
}

bool Matrix::operator=(const Matrix& mal){
    this->setSize(mal.size);
    for(int i = 0 ; i < size.x; i++)
        memcpy(data[i], mal.data[i], size.y * sizeof(double) );
    return true;
}

Matrix::~Matrix()
{
    if (data == NULL || size.x == 0 || size.y == 0)
        return;

    if(oneblock == true){
        free(data[0]);
        free(data);
        return;
    }
    for(int i = 0 ; i < size.x; i++){
        free(data[i]);
    }
    free(data);
}

void Matrix::apply(double(*func)(double)){
    for(int i = 0 ; i < size.x ; i ++){
        for(int k = 0 ; k <size.y;k++){
            data[i][k] = func(data[i][k]);
        }
    }
}

void Matrix::setSize(int x,int y,bool oneblock){
    if(x*y == size.x * size.y && this->oneblock == true){
        size.x = x;
        size.y = y;
        //re_assign address; 
        for(int i = 1 ; i < size.x ; i++){
            data[i] = data[i-1] + size.y; // this is legit :))
        }
        return;
    }
    this->~Matrix();
    this->oneblock = oneblock;
    this->size.x = x;
    this->size.y = y;

    this->data = (double**)malloc(size.x*sizeof(double*));

    if (oneblock == true){
        data[0] = (double*)malloc(size.x*size.y*sizeof(double));
        if(data[0] != NULL){
            for(int i = 1 ; i < size.x ; i++){
                data[i] = data[i-1] + size.y; // this is legit :))
            }
            return;
        }
        this->oneblock = false;
    }
    for(int i = 0; i < size.x ; i++){
        data[i] = (double*)malloc(size.y*sizeof(double));
    }
}

//operator

double*& Matrix::operator[](int idx){
    return this->data[idx];
}


bool Matrix::operator+=(double v){
    for(int i = 0 ; i < size.x ; i ++){
        for(int k = 0 ; k < size.y;k++){
            this->data[i][k] += v;
        }
    }
    return true;
}

bool Matrix::operator-=(double v){
    return this->operator+=(-v);
}

bool Matrix::operator*=(double v){
    for(int i = 0 ; i < size.x ; i ++){
        for(int k = 0 ; k < size.y;k++){
            this->data[i][k] *= v;
        }
    }
    return true;
}

bool Matrix::operator/=(double v){
    return this->operator*=(1/v);
}

bool Matrix::operator+=(const Matrix& m){
    if (m.size.x == 0 || m.size.y == 0)
        return true;
    if( size.x == m.size.x){
        bool is_col = !(m.size.y == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] += m.data[i][k*is_col];
            }
        }
        return true;
    }
    else if (size.y == m.size.y){
        bool is_row = !(m.size.x == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] += m.data[i*is_row][k];
            }
        }
        return true;
    }
    return false;
}

bool Matrix::operator-=(const Matrix& m){
    if (m.size.x == 0 || m.size.y == 0)
        return true;
    if( size.x == m.size.x){
        bool is_col = !(m.size.y == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] -= m.data[i][k*is_col];
            }
        }
        return true;
    }
    else if (size.y == m.size.y){
        bool is_row = !(m.size.x == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] -= m.data[i*is_row][k];
            }
        }
        return true;
    }
    return false;
}

bool Matrix::operator*=(const Matrix& m){
    if (m.size.x == 0 || m.size.y == 0)
        return true;
    if( size.x == m.size.x){
        bool is_col = !(m.size.y == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] *= m.data[i][k*is_col];
            }
        }
        return true;
    }
    else if (size.y == m.size.y){
        bool is_row = !(m.size.x == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] *= m.data[i*is_row][k];
            }
        }
        return true;
    }
    return false;
}

bool Matrix::operator/=(const Matrix& m){
    if (m.size.x == 0 || m.size.y == 0)
        return true;
    if( size.x == m.size.x){
        bool is_col = !(m.size.y == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] /= m.data[i][k*is_col];
            }
        }
        return true;
    }
    else if (size.y == m.size.y){
        bool is_row = !(m.size.x == 1);
        for(int i = 0 ; i < size.x; i++){
            for(int k = 0 ; k < size.y ; k ++){
                data[i][k] /= m.data[i*is_row][k];
            }
        }
        return true;
    }
    return false;
}

void Matrix::ToOneHot(int num_catagory){
    Matrix temp(size.x,num_catagory);
    temp.setval(0);
    for(int i = 0 ; i < size.x ; i++){
        temp.data[i][(int)data[i][0]] = 1;
    }
    std::swap(temp.size,this->size);
    std::swap(temp.data,this->data);
    std::swap(temp.oneblock,this->oneblock);
}

void Matrix::setval(int val){
    if(oneblock == true){
        memset(data[0],val,sizeof(double)*size.x*size.y);
        return;
    }
    for(int i = 0 ; i < size.x; i++){
        memset(data[i],val,sizeof(double)*size.y);
    }
}