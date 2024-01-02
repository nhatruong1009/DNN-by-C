#ifndef __MATRIX__H__
#define __MATRIX__H__
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <time.h>

class Matrix
{
private:
    struct dim
    {
        int x;
        int y;
    };  
    bool oneblock;
public:
    //
    double** data;
    dim size;
    //
    Matrix();
    Matrix(int x,int y,bool oneblock = true);
    Matrix(const Matrix& mal);
    ~Matrix();
    void setval(int val);
    void ToOneHot(int num_catagory);
    void apply(double(*func)(double));

    void setSize(int x,int y,bool oneblock = true);
    void setSize(dim _size,bool _oneblock = true){
        setSize(_size.x,_size.y,_oneblock);
    }
    bool resize(int x,int y){
        if(x*y !=  size.x * size.y)
            return false;
        if(!this->is_one_block())
            return false;

        size.x = x;
        size.y = y;

        for(int i  = 1; i < size.x ; i ++){
            this->data[i] = data[i-1] + size.y;
        }
        return true;
    }


    //remove later

    void printSize(){
        printf("[%d %d]\n",size.x,size.y);
    }
    void printSize(FILE *&fil){
        fprintf(fil,"[%d %d]\n",size.x,size.y);
    }

    void update_one_block_status(){
        oneblock = true;
        if (size.x < 2)
            return;
        int s = data[1] - data[0];
        for(int i = 2 ; i < size.x ; i ++){
            if (data[i] - data[i-1] != s){
                oneblock = false;
                return;
            }
        }
    }

    bool is_one_block() const {
        return oneblock;
    }

    
    void print(bool _int = false) const{
        if (!_int)
        for(int i = 0 ; i < size.x ; i ++){
            for (int j = 0 ; j < size.y; j ++){
                printf("%2.4lf ",data[i][j]);
            }
            printf("\n");
        }
        else
        for(int i = 0 ; i < size.x ; i ++){
            for (int j = 0 ; j < size.y; j ++){
                printf("%d\t",(int)data[i][j]);
            }
            printf("\n");
        }
    }
    

    // operator

    bool operator=(const Matrix& m);

    double*& operator[](int idx);
    bool operator+=(double v);
    bool operator-=(double v);
    bool operator*=(double v);
    bool operator/=(double v);

    bool operator+=(const Matrix& m);
    bool operator-=(const Matrix& m);
    bool operator*=(const Matrix& m);
    bool operator/=(const Matrix& m);

    friend void swap(Matrix &a, Matrix&b){
        std::swap(a.data,b.data);
        std::swap(a.size,b.size);
        std::swap(a.oneblock,b.oneblock);
    }

};

class dataBatch{
    public:
    Matrix* data;
    Matrix* labels;
    int batch_size;
    int n_batch;
    dataBatch(const Matrix& _data, const Matrix& _labels,int batch_size);
    ~dataBatch();
};

#endif