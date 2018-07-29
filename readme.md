# Resnet with Keras.

This project is about building Resnet with Keras
Deep network degradation is occured so Resnet adopted residual block.
resnet.py build network and train.py train the model then test it.

https://arxiv.org/pdf/1512.03385.pdf

## Requirements

This project use only keras with tensorflow backend.
So you just install keras and tensorflow.

## Dataset

Keras provide easy way to use dataset.
cifar10 is quite small dataset for Resnet-50.
But it's focused on effect of residual network for address degradation of deep network.
So cifar10 is used for convinient. Actually small CNN network is proper to cifar10.

## Training

Below two images are accuracy of resnet-50 on cifar10.
As you can see non residual model is not well trained.
But Residual network is trained well.

- Orange: With short connection
- Blue: Without short connection

![alt text](https://github.com/Sangkwun/Resnet/blob/master/training.png)

![alt text](https://github.com/Sangkwun/Resnet/blob/master/validating.png)

## Conclusion

Two model has only difference of residual block existence.
By comparison on two model short connection make deeper network can be trained well.
