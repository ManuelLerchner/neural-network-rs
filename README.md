![Build-Status](https://github.com/ManuelLerchner/neural-network-rs/actions/workflows/rust.yml/badge.svg)
[![Rust](https://img.shields.io/badge/rust-1.59.0%2B-blue.svg?maxAge=3600)](https://github.com/rust-lang/regex)
# Neural-Network

A simple neural network written in rust.

## About

This implementation of a neural network using gradient-descent is completely written from ground up using rust.
It is possible to specify the shape of the network, as well as the learning-rate of the network. Additionally, you can choose from one of many predefined datasets, for example the XOR- and CIRCLE Datasets, which represent the relative functions inside the union-square. As well as more complicated datasets like the RGB_DONUT, which represents a donut-like shape with a rainbow like color transition.

Below, you can see a training process, where the network is trying to learn the color-values of the RGB_DONUT dataset.

## Features

The following features are currently implemented:

- **Optimizers**
  1. Adam
  2. RMSProp
  3. SGD
- **Loss Functions**
  1. Quadratic
- **Activation Functions**
  1. Sigmoid
  2. ReLU
- **Layers**
  1. Dense
- **Plotting**
  1. Plotting the cost-history during training
  2. Plotting the final predictions inside, either in grayscale or RGB

## Usage

The process of creating and training the neural network is
pretty straightforwards:

![carbon](https://user-images.githubusercontent.com/54124311/195871975-e211c2b7-d055-4cb5-852f-bf9e031a3aab.png)

## Example Training Process

Below, you can see how the network learns:

### Learning Animation

<https://user-images.githubusercontent.com/54124311/195410077-7a02b075-0269-4ff2-965f-97f224ab2cf1.mp4>

### Final Result

![RGB_DONUT_SGD_ 2,64,64,64,64,64,3](https://user-images.githubusercontent.com/54124311/195409668-7db568af-9232-489b-a149-108d63c8d23a.png)

## Cool training results

### RGB_DONUT

#### Big Network

![RGB_DONUT_RMS_PROP_ 2,128,128,128,3](https://user-images.githubusercontent.com/54124311/195874011-29deb232-1640-4bdd-9c19-02f8e4fed5b8.png)
![RGB_DONUT_RMS_PROP_ 2,128,128,128,3 _history](https://user-images.githubusercontent.com/54124311/195874028-3513e6f4-fc2c-492f-81f0-ef64fd96e176.png)

#### Small Network

![RGB_DONUT_SGD_ 2,8,8,8,3](https://user-images.githubusercontent.com/54124311/195876194-cf42fcf6-ba93-4151-acb2-fd945effe824.png)
![RGB_DONUT_SGD_ 2,8,8,8,3 _history](https://user-images.githubusercontent.com/54124311/195876200-420bbaaf-5587-490c-90cb-2f29ca11278b.png)

### XOR_PROBLEM

![XOR_SGD_ 2,8,8,8,1](https://user-images.githubusercontent.com/54124311/195877107-4c8dbca5-bc93-4e14-a0b2-a27bcc47a3f5.png)
![XOR_SGD_ 2,8,8,8,1 _history](https://user-images.githubusercontent.com/54124311/195877118-03cbd210-f34f-4bc9-ac22-e870a74e9e0d.png)
