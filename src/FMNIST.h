#ifndef SRC_FMNIST_H_
#define SRC_FMNIST_H_

#include <fstream>
#include <iostream>
#include <string>
#include "matrix.h"

class FMNIST {
 private:
  std::string data_dir;

 public:
  Matrix train_data;
  Matrix train_labels;
  Matrix test_data;
  Matrix test_labels;

  void read_fmnist_data(std::string filename,Matrix& data);
  void read_fmnist_label(std::string filename,Matrix& labels);

  explicit FMNIST(std::string data_dir) : data_dir(data_dir) {  }
  void read();
  void dataInfo(){
    printf("train data  : "); train_data.printSize();
    printf("train labels: "); train_labels.printSize();
    printf("test data   : "); test_data.printSize();
    printf("test labels : "); test_labels.printSize();
  }
};

#endif  // SRC_MNIST_H_
