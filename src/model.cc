#include "model.h"

void Model::add(Layer &layer){
    layers.push_back(&layer);
}

void Model::forward(const Matrix& input){
    layers[0]->forward(input);
    for(int i = 1 ; i < layers.size();i++){
        layers[i]->forward(layers[i-1]->_foward);
    }
    this->_foward = layers[layers.size()-1]->_foward;
}

void Model::backward(const Matrix& _input,const Matrix& _target){
    this->loss_layer->evaluate(this->_foward,_target);    
    layers[layers.size()-1]->backward(layers[layers.size()-2]->_foward,loss_layer->back_gradient());

    for(int i = layers.size()-2; i > 0 ; i--){
        layers[i]->backward(layers[i-1]->_foward,layers[i+1]->_backward);
    }

    layers[0]->backward(_input,layers[1]->_backward);
}

void Model::update(Optimizer& opt){
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->update(opt);
    }
}