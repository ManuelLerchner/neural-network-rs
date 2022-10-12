# Neural-Network
A simple neural network written in rust.


## About
This implementation of a neural network using gradient-descent is completely written from ground up using rust.
It is possible to specify the shape of the network, as well as  the learning-rate of the network. Additionally, you can choose from one of many predefined datasets, for example the XOR- and CIRCLE Datasets, which represent the relative functions inside the union-square. As well as more complicated datasets like the RGB_DONUT, which represents a donut-like shape with a rainbow like color transition.

Below, you can see a training process, where the network is trying to learn the color-values of the RGB_DONUT dataset.

## Example Training Process

### Training-Progress
![RGB_DONUT_SGD_ 2,64,64,64,64,64,3 _history](https://user-images.githubusercontent.com/54124311/195409972-a8278be2-4a2b-4bcf-9375-79ab594831ed.png)

### Learning Animation
https://user-images.githubusercontent.com/54124311/195410077-7a02b075-0269-4ff2-965f-97f224ab2cf1.mp4


### Final Result
![RGB_DONUT_SGD_ 2,64,64,64,64,64,3](https://user-images.githubusercontent.com/54124311/195409668-7db568af-9232-489b-a149-108d63c8d23a.png)
