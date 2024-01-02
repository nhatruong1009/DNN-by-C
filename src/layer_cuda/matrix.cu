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


#ifdef UnifiedMem //

Matrix::Matrix(int x,int y,bool oneblock){   
    this->size.x = x;
    this->size.y = y;
    this->oneblock = oneblock;
    CHECK(cudaMallocManaged(&this->data,sizeof(double*)*size.x));
    if (oneblock == true){
        CHECK(cudaMallocManaged(&this->data[0],sizeof(double)*size.x*size.y));
        if(data[0] != NULL){
            for(int i = 1 ; i < size.x ; i++){
                data[i] = data[0] + size.y*i; // this is legit :))
            }
            return;
        }
        this->oneblock = false;
    }
    
    for(int i = 0; i < size.x ; i++)
        CHECK(cudaMallocManaged(&this->data[i],sizeof(double)*size.y));
}

Matrix::Matrix(const Matrix& mal){
    this->setSize(mal.size);

    if (this->is_one_block() && mal.is_one_block()){
        memcpy(data[0],mal.data[0],size.x * size.y * sizeof(double));
        //CHECK(cudaMemcpy(data[0],mal.data[0],size.x * size.y * sizeof(double),cudaMemcpyDefault));
        return;
    }

    for(int i = 0 ; i < size.x; i++){
        memcpy(data[i],mal.data[i],size.y * sizeof(double));
        //CHECK(cudaMemcpy(data[i],mal.data[i],size.y * sizeof(double),cudaMemcpyDefault));
    }
}

bool Matrix::operator=(const Matrix& mal){
    this->setSize(mal.size);

    if (this->is_one_block() && mal.is_one_block()){
        memcpy(data[0],mal.data[0],size.x * size.y * sizeof(double));
        //CHECK(cudaMemcpy(data[0],mal.data[0],size.x * size.y * sizeof(double),cudaMemcpyDefault));
        return true;
    }

    for(int i = 0 ; i < size.x; i++){
        memcpy(data[i],mal.data[i],size.y * sizeof(double));
        //CHECK(cudaMemcpy(data[0],mal.data[0], size.y * sizeof(double),cudaMemcpyDefault));
    }
    return true;
}

Matrix::~Matrix()
{
    if (data == NULL || size.x == 0 || size.y == 0)
        return;

    if(oneblock == true){
        cudaFree(data[0]);
        cudaFree(data);
        return;
    }
    
    for(int i = 0 ; i < size.x; i++)
        cudaFree(data[i]);
    
    cudaFree(data);
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
    
    CHECK(cudaMallocManaged(&this->data,sizeof(double*)*size.x));

    if (oneblock == true){
        CHECK(cudaMallocManaged(&this->data[0],sizeof(double*)*size.x*size.y));
        if(data[0] != NULL){
            for(int i = 1 ; i < size.x ; i++){
                data[i] = data[i-1] + size.y; // this is legit :))
            }
            return;
        }
        this->oneblock = false;
    }
    
    for(int i = 0; i < size.x ; i++)
        CHECK(cudaMallocManaged(&this->data[i],sizeof(double*)*size.y));
}


#else // Ram mem

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
    
    for(int i = 0; i < size.x ; i++)
        data[i] = (double*)malloc(size.y*sizeof(double));
}

Matrix::Matrix(const Matrix& mal){
    this->setSize(mal.size);

    if (this->is_one_block() && mal.is_one_block()){
        memcpy(data[0], mal.data[0], size.x * size.y * sizeof(double) );
        return;
    }

    for(int i = 0 ; i < size.x; i++){
        memcpy(data[i], mal.data[i], size.y *sizeof(double));
    }
}

bool Matrix::operator=(const Matrix& mal){
    this->setSize(mal.size);

    if (this->is_one_block() && mal.is_one_block()){
        memcpy(data[0], mal.data[0], size.x * size.y * sizeof(double) );
        return true;
    }

    for(int i = 0 ; i < size.x; i++){
        memcpy(data[i], mal.data[i], size.y * sizeof(double) );
    }
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
    
    for(int i = 0 ; i < size.x; i++)
        free(data[i]);
    
    free(data);
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
    
    for(int i = 0; i < size.x ; i++)
        data[i] = (double*)malloc(size.y*sizeof(double));
}


#endif


Matrix::Matrix(){
    this->size.x = 0;
    this->size.y = 0;
    this->data = NULL;
}



void Matrix::apply(double(*func)(double)){
    for(int i = 0 ; i < size.x ; i ++){
        for(int k = 0 ; k <size.y;k++){
            data[i][k] = func(data[i][k]);
        }
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


#ifdef UnifiedMem
    dataBatch::dataBatch(const Matrix& _data, const Matrix& _labels,int batch_size){
        this->batch_size = batch_size;
        n_batch = (_data.size.x-1)/batch_size+1;

        data = (Matrix*)malloc(n_batch*sizeof(Matrix));
        labels = (Matrix*)malloc(n_batch*sizeof(Matrix));

        double** arr;
        cudaMallocManaged(&arr,_data.size.x*sizeof(double*));
        double** arr_labels;
        cudaMallocManaged(&arr_labels,_data.size.x*sizeof(double*));

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
            data[i].data;
            cudaMallocManaged(&data[i].data,size*sizeof(double*));
            data[i].size.x = size;
            data[i].size.y = _data.size.y;
            memcpy(data[i].data,&arr[i*batch_size],size*sizeof(double*));
            
            labels[i].data;
            cudaMallocManaged(&labels[i].data,size*sizeof(double*));
            labels[i].size.x = size;
            labels[i].size.y = _labels.size.y;
            memcpy(labels[i].data,&arr_labels[i*batch_size],size*sizeof(double*));
        }
        cudaFree(arr);
        cudaFree(arr_labels);
        data->update_one_block_status();
        labels->update_one_block_status();
    }
    dataBatch::~dataBatch(){
        for(int i = 0 ; i < n_batch;i++){
            cudaFree(data[i].data);
            cudaFree(labels[i].data);
            data[i].data=NULL;
            labels[i].data = NULL;
        }
        free(data);
        free(labels);
    }


#else

dataBatch::dataBatch(const Matrix& _data, const Matrix& _labels,int batch_size){
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
    data->update_one_block_status();
    labels->update_one_block_status();
}
dataBatch::~dataBatch(){
    for(int i = 0 ; i < n_batch;i++){
        free(data[i].data);
        free(labels[i].data);
        data[i].data=NULL;
        labels[i].data = NULL;
    }
    free(data);
    free(labels);
}

#endif