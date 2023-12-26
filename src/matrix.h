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
    dataBatch(const Matrix& _data, const Matrix& _labels,int batch_size){
        this->batch_size = batch_size;
        n_batch = (_data.size.x-1)/batch_size+1;

        data = (Matrix*)malloc(n_batch*sizeof(Matrix));
        labels = (Matrix*)malloc(n_batch*sizeof(Matrix));

        double** arr = (double**)malloc(_data.size.x*sizeof(double*));
        double** arr_labels = (double**)malloc(_data.size.x*sizeof(double*));

        memcpy(arr,_data.data,_data.size.x*sizeof(double*));
        memcpy(arr_labels,_labels.data,_data.size.x*sizeof(double*));
        // shuffer

        srand ( time(NULL) );
 
        for (int i = _data.size.x-1; i > 0; i--)
        {
            int j = rand() % (i+1);
            std::swap(arr[i], arr[j]);
            std::swap(arr_labels[i], arr_labels[j]);
        }

        for(int i = 0 ; i < n_batch; i ++){

            int size = (i != n_batch-1) ? batch_size : _data.size.x - i*batch_size;
            data[i].data = (double**)malloc(size*sizeof(double*));
            data[i].size.x = size;
            data[i].size.y = _data.size.y;
            memcpy(data[i].data,&arr[i*batch_size],size*sizeof(double*));
            
            labels[i].data = (double**)malloc(size*sizeof(double*));
            labels[i].size.x = size;
            labels[i].size.y = _labels.size.y;
            memcpy(labels[i].data,&arr_labels[i*batch_size],size*sizeof(double*));

        
        }
        free(arr);
        free(arr_labels);
    }
    ~dataBatch(){
        for(int i = 0 ; i < n_batch;i++){
            free(data[i].data);
            free(labels[i].data);
            data[i].data=NULL;
            labels[i].data = NULL;
        }
        free(data);
        free(labels);
    }
};


#endif