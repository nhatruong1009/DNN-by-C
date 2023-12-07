#include "FMNIST.h"

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void FMNIST::read_fmnist_data(std::string filename,Matrix& data) {
  std::ifstream file(filename, std::ios::binary);
  //Matrix *data;
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    unsigned char label;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);

    data.setSize(number_of_images,n_rows*n_cols);
    for (int i = 0; i < number_of_images; i++) {
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char image = 0;
          file.read((char*)&image, sizeof(image));
          data[i][r * n_cols + c] = (double)image;
        }
      }
    }
  }
}

void FMNIST::read_fmnist_label(std::string filename,Matrix& labels) {
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);

    labels.setSize(number_of_images,1);
    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;
      file.read((char*)&label, sizeof(label));
      labels[i][0] = (float)label;
    }
  }
}

void FMNIST::read() {
  read_fmnist_data(data_dir + "train_data",this->train_data);
  read_fmnist_data(data_dir + "test_data",this->test_data);
  read_fmnist_label(data_dir + "train_label",this->train_labels);
  read_fmnist_label(data_dir + "test_label",this->test_labels);
}
