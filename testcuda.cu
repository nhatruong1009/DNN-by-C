#include "src/matrix.h"
#include "src/layer_cuda/CNN_v1.h"
#include "src/layer_cuda/CNN_v2.h"
#include "src/layer_cuda/CNN_v3.h"
#include "src/layer/convolution.h"
#include "src/matrix_math.h"
#include <math.h>


struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};



void compare(Matrix&a, Matrix&b){
    double sum = 0;
    for(int i = 0 ; i < a.size.x ; i++){
        for(int j = 0 ; j < a.size.y ; j ++){
            sum += abs(a[i][j] - b[i][j]); 
        }
    }
    sum /= (a.size.x * a.size.y);
    printf("error: %lf\n", sum);
}


int main(){
  
    Matrix a;
    a.setSize(512,28*28);
  
    set_normal_random_matrix(a,240,10);
    Convolution cnn = Convolution(1,28,28,3,5,false);
    CNN_cuda_v1 conv1 = CNN_cuda_v1(1,28,28,3,5,false);
    CNN_cuda_v2 conv2 = CNN_cuda_v2(1,28,28,3,5,false);
    CNN_cuda_v3 conv3 = CNN_cuda_v3(1,28,28,3,5,false);
    
    conv1.filter = cnn.filter;
    conv1.bias = cnn.bias;

    conv2.filter = cnn.filter;
    conv2.bias = cnn.bias;

    conv3.filter = cnn.filter;
    conv3.bias = cnn.bias;

    GpuTimer timer; 
    int loop = 10;

    
    timer.Start();
    for(int i = 0 ; i < loop ; i ++)
      cnn.forward(a);
    timer.Stop();
    printf("host time: %.3f ms\n", timer.Elapsed());

     
    timer.Start();
    for(int i = 0 ; i < loop ; i ++)
      conv1.forward(a);
    timer.Stop();
    printf("device1 time: %.3f ms\n", timer.Elapsed());
    compare(cnn._foward,conv1._foward);

    timer.Start();
    for(int i = 0 ; i < loop ; i ++)
      conv2.forward(a);
    timer.Stop();
    printf("device2 time: %.3f ms\n", timer.Elapsed());
    compare(cnn._foward,conv2._foward);

    timer.Start();
    for(int i = 0 ; i < loop ; i ++)
      conv3.forward(a);
    timer.Stop();
    printf("device3 time: %.3f ms\n", timer.Elapsed());
    compare(cnn._foward,conv3._foward);
    
    


    return 0;
}