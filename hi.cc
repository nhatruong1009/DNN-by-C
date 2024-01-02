#include "src/matrix.h"
#include "src/matrix_math.h"
#include "src/FMNIST.h"
#include "src/model.h"
#include "src/layer/fullyconnected.h"
#include "src/layer/ReLU.h"
#include "src/layer/sigmoid.h"
#include "src/layer/maxPooling.h"
#include "src/layer/avgPooling.h"
#include "src/layer/convolution.h"
#include "src/matrix_math.h"
#include "src/loss.h"
#include "src/optimizer/sgd.h"
#include "src/layer/softmax.h"

#include <time.h>
clock_t start, end;
double cpu_time_used;

int main(){
    FMNIST data("./FMNIST/");
    printf("read data\n");
    data.read();
    printf("read done\n");
    data.dataInfo();

    data.test_labels.ToOneHot(10);
    data.train_labels.ToOneHot(10);


    Convolution conv1 = Convolution(1,28,28,6,5,true);
    Avg_pooling pool1 = Avg_pooling(6,28,28,2,2);
    Convolution conv2 = Convolution(6,14,14,16,5,false);
    Avg_pooling pool2 = Avg_pooling(16,10,10,2,2);

    FullyConnected full1 = FullyConnected(400,120);
    FullyConnected full2 = FullyConnected(120,32);
    FullyConnected full3 = FullyConnected(32,10);
    
    
    ReLU relu1;
    ReLU relu2;
    ReLU relu3;
    ReLU relu4;
    Softmax softmax;
    
    Model model;

    model.add(conv1); //28 x28 x3
    model.add(relu1);
    model.add(pool1); // 14x14x3
    model.add(conv2); //
    model.add(relu2); 
    model.add(pool2);

    model.add(full1);
    model.add(relu3);
    model.add(full2);
    model.add(relu4);
    
    model.add(full3);
    model.add(softmax);

    Cross_entropy temp;
    model.addloss(temp);

    SGD opt(0.002, 5e-4, 0.9, true);
    //return 0;
    //


    int n_epoch = 1;
    int batch_size = 128;
    double time_avg = 0;
    int count = 1;
    for(int epoch = 0 ; epoch < n_epoch ; epoch++){
        printf("epoch: %d \n",epoch);
        dataBatch data_batch(data.train_data,data.train_labels,batch_size);
        for(int idx = 0 ; idx < data_batch.n_batch ; idx++){

            start = clock();

            model.forward(data_batch.data[idx]);
            model.backward(data_batch.data[idx],data_batch.labels[idx]);          
            model.update(opt);

    
            end = clock();
            time_avg = (time_avg*count + ((double) (end - start)) / CLOCKS_PER_SEC)/(count+1);
            count+=1;
            if (idx % 50 == 0){
                printf("\t%d-th: loss: %lf \t time executed avg: %lf (s/batch)\n",idx,model.loss_layer->output(),time_avg);    
            }
        }
        model.forward(data.test_data);
        printf("accuracy:%lf\n",compute_accuracy(model._foward,data.test_labels));
        model.forward(data_batch.data[0]);
        model._foward.print();
    }
    return 0;

}