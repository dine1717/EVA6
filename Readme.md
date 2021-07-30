Spatial Transformer


A Spatial Transformer is an image model block that explicitly allows the spatial manipulation of data within a convolutional neural network. It gives CNNs the ability to actively spatially transform feature maps, conditional on the feature map itself, without any extra training supervision or modification to the optimisation process. Unlike pooling layers, where the receptive fields are fixed and local, the spatial transformer module is a dynamic mechanism that can actively spatially transform an image (or a feature map) by producing an appropriate transformation for each input sample. The transformation is then performed on the entire feature map (non-locally) and can include scaling, cropping, rotations, as well as non-rigid deformations.

![image](https://user-images.githubusercontent.com/73247157/127658957-c9d5e050-22c4-4604-9981-4ad660d2eac8.png)


Spatial transformer networks boils down to three main components :

1. The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
2. The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
3. The sampler uses the parameters of the transformation and applies it to the input image.


