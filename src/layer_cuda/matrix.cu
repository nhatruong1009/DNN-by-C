#include "../matrix.h"
#include <cstring>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}



Matrix::Matrix(){
    this->size.x = 0;
    this->size.y = 0;
    this->data = NULL;
}

Matrix::Matrix(int x,int y,bool oneblock){   
    this->size.x = x;
    this->size.y = y;
    this->oneblock = oneblock;
    #ifndef UnifiedMem
        this->data = (double**)malloc(size.x*sizeof(double*));
    #else
        CHECK(cudaMallocManaged(&this->data,sizeof(double*)*size.x));
    #endif
    if (oneblock == true){
        #ifndef UnifiedMem
            data[0] = (double*)malloc(size.x*size.y*sizeof(double));
        #else
            CHECK(cudaMallocManaged(&this->data[0],sizeof(double)*size.x*size.y));
        #endif
        if(data[0] != NULL){
            for(int i = 1 ; i < size.x ; i++){
                data[i] = data[0] + size.y*i; // this is legit :))
            }
            return;
        }
        this->oneblock = false;
    }
    
    #ifndef UnifiedMem
        for(int i = 0; i < size.x ; i++)
            data[i] = (double*)malloc(size.y*sizeof(double));
    #else
        for(int i = 0; i < size.x ; i++)
            CHECK(cudaMallocManaged(&this->data[i],sizeof(double)*size.y));
    #endif
}

Matrix::Matrix(const Matrix& mal){
    this->setSize(mal.size);

    if (this->is_one_block() && mal.is_one_block()){
        #ifndef UnifiedMem
            memcpy(data[0], mal.data[0], size.x * size.y * sizeof(double) );
        #else
            CHECK(cudaMemcpy(data[0],mal.data[0],size.x * size.y * sizeof(double),cudaMemcpyDefault));
        #endif
        return;
    }

    for(int i = 0 ; i < size.x; i++){
        #ifndef UnifiedMem
            memcpy(data[i], mal.data[i], size.y *sizeof(double));
        #else
            CHECK(cudaMemcpy(data[i],mal.data[i],size.y * sizeof(double),cudaMemcpyDefault));
        #endif
    }
}

bool Matrix::operator=(const Matrix& mal){
    this->setSize(mal.size);

    if (this->is_one_block() && mal.is_one_block()){
        #ifndef UnifiedMem
            memcpy(data[0], mal.data[0], size.x * size.y * sizeof(double) );
        #else
            CHECK(cudaMemcpy(data[0],mal.data[0],size.x * size.y * sizeof(double),cudaMemcpyDefault));
        #endif
        return true;
    }

    for(int i = 0 ; i < size.x; i++){
        #ifndef UnifiedMem
            memcpy(data[i], mal.data[i], size.y * sizeof(double) );
        #else
            CHECK(cudaMemcpy(data[0],mal.data[0], size.y * sizeof(double),cudaMemcpyDefault));
        #endif
    }
    return true;
}

Matrix::~Matrix()
{
    if (data == NULL || size.x == 0 || size.y == 0)
        return;

    if(oneblock == true){
        #ifndef UnifiedMem
            free(data[0]);
            free(data);
        #else
            cudaFree(data[0]);
            cudaFree(data);
        #endif
        return;
    }
    
    #ifndef UnifiedMem
        for(int i = 0 ; i < size.x; i++)
            free(data[i]);
    #else
        for(int i = 0 ; i < size.x; i++)
            cudaFree(data[i]);
    #endif
    
    #ifndef UnifiedMem
        free(data);
    #else
        cudaFree(data);
    #endif
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
    
    #ifndef UnifiedMem
        this->data = (double**)malloc(size.x*sizeof(double*));
    #else
        CHECK(cudaMallocManaged(&this->data,sizeof(double*)*size.x));
    #endif

    if (oneblock == true){
        #ifndef UnifiedMem
            data[0] = (double*)malloc(size.x*size.y*sizeof(double));
        #else
            CHECK(cudaMallocManaged(&this->data[0],sizeof(double*)*size.x*size.y));
        #endif
        if(data[0] != NULL){
            for(int i = 1 ; i < size.x ; i++){
                data[i] = data[i-1] + size.y; // this is legit :))
            }
            return;
        }
        this->oneblock = false;
    }
    
    #ifndef UnifiedMem
        for(int i = 0; i < size.x ; i++)
            data[i] = (double*)malloc(size.y*sizeof(double));
    #else
        for(int i = 0; i < size.x ; i++)
            CHECK(cudaMallocManaged(&this->data[i],sizeof(double*)*size.y));
    #endif
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
        #ifndef UnifiedMem
            memset(data[0],val,sizeof(double)*size.x*size.y);
        #else
            CHECK(cudaMemset(data[0],val,sizeof(double)*size.x*size.y));
        #endif
        return;
    }
    #ifndef UnifiedMem
        for(int i = 0 ; i < size.x; i++)
        memset(data[i],val,sizeof(double)*size.y);
    #else
        for(int i = 0 ; i < size.x; i++)
            CHECK(cudaMemset(data[i],val,sizeof(double)*size.y));
    #endif

}