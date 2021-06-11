# Different types of Normalization <br>

Group:
* Bharath
* Sabeesh
* Dinesh
* Manu
#### Explanation of code

1) The model used here is the the model trained on the MNIT dataset for the prediction of the digits in the dataset
2) model.py file consists of three different models built using three different normalization techniques - Batch normalization, Layer normalization and Group normalization.
3) Three classes are defined inside the model.py file. 
4) Pytorch  nn module has been used to implement the <br>

    (a) BatchNorm2d() where batch normalization is done for each channel in a particular layer across all the batches. Hence the number of parameters used here would be 
    2 * number of channels (1 gamma and 1 beta for each channel across all imgaes of that particular layer in that batch)<br>
    
    (b) Layer normalization is done using GroupNorm(1, channels) -> which indicates that there is one group across all the channels and sonormalization is done across all 
    the channels in that particular layer of the image. hence there are only two paramters  here technically (1 gamma and 1 beta). This is done by using the function 
    Groupnorm(n_groups =1, n_channels)<br>
    
    (c) Group normalization is also done horizontally across all the channels in a layer which is divided into n groups. So if there are two groups (16 channels) there should        be two normalizations taking place (8 channels each) which each of it having 1 gamma and 1 beta. Thisis done by using the function Groupnorm(n_groups, n_channels)<br>
    
    The Pytorch documentation:<br>
    
    Unlike as mentioned above, Pytorch implements group normalization and layer normalization by having one gamma and one beta alloted for each of the channels. Hence the total number of paramters reamin the same for group layer and batch normalization as the gamma and beta are separate for each channel. However the normalization is done as per the concepts.<br> 
    
      - Batchnormalization does normalization for each channel across batches for a particular layer. (Vertical)<br>
      - Layer normalization does normalization for all channels in a layer for an image. (Horizontal)<br>
      - Group normalization does normalization for a group of channels in a layer for an image (Horizontal)<br>
5) L1 loss is implemented in the training function and the l1 loss is added to the total loss before doing backpropogation or differentiation.<br>

 
#### Explanation of Excel sheet <br>

1) As mentioned above in case of Batch normalization, the values across all the channels (vertical) in a particular layer for all the images  in a batch are normalized and then the shift and scale is reapplied (Gamma and beta) for EACH CHANNEL. <br>

![batchnorm](https://user-images.githubusercontent.com/84949894/121703687-b4d8c900-caf0-11eb-9fec-0764589904df.PNG) <br>

2) In case of layer normalization (black arrow), the value across all the channels in a layer of a particular image are normalized amongst itself (horizontal). There will be one gamma and one beta for each channel <br>
![group and layer](https://user-images.githubusercontent.com/84949894/121703762-c6ba6c00-caf0-11eb-8a98-1042af0cb11c.PNG) <br>
4) In case of group normalization (blue arrow) as shown above the channels are divided into corresponding groups and the groups are normalized separately. Each channel here will again have a gamma and a beta as the trainable parameters. <br>
5) The explanation of the gammma and the beta in the normalization technique is as mentioned below. They can shift an image anywhere between completely normalized to the original image and anywhere in between as shown below. <br>
![gamma_beta](https://user-images.githubusercontent.com/84949894/121705563-62001100-caf2-11eb-89ce-c8ad93fe55e5.jpeg) <br>

### Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
            Conv2d-4           [-1, 12, 24, 24]             864
              ReLU-5           [-1, 12, 24, 24]               0
       BatchNorm2d-6           [-1, 12, 24, 24]              24
            Conv2d-7            [-1, 8, 24, 24]              96
         MaxPool2d-8            [-1, 8, 12, 12]               0
            Conv2d-9           [-1, 12, 10, 10]             864
             ReLU-10           [-1, 12, 10, 10]               0
      BatchNorm2d-11           [-1, 12, 10, 10]              24
           Conv2d-12             [-1, 16, 8, 8]           1,728
             ReLU-13             [-1, 16, 8, 8]               0
      BatchNorm2d-14             [-1, 16, 8, 8]              32
           Conv2d-15             [-1, 12, 8, 8]             192
        MaxPool2d-16             [-1, 12, 4, 4]               0
           Conv2d-17             [-1, 15, 4, 4]           1,620
             ReLU-18             [-1, 15, 4, 4]               0
      BatchNorm2d-19             [-1, 15, 4, 4]              30
           Conv2d-20             [-1, 15, 4, 4]           2,025
             ReLU-21             [-1, 15, 4, 4]               0
      BatchNorm2d-22             [-1, 15, 4, 4]              30
        AvgPool2d-23             [-1, 15, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             150
================================================================
Total params: 7,767
```

### Explained Pictorially
![image](https://user-images.githubusercontent.com/10822997/121735397-b4055e80-cb13-11eb-9c20-355970906925.png)
* The above image presented in the research paper is one of the best ways to compare the various normalization techniques and get an intuitive understanding for GN.
* Let’s consider that we have a batch of dimension (N, C, H, W) that needs to be normalized.

N: Batch Size
C: Number of Channels
H: Height of the feature map
W: Width of the feature map
https://amaarora.github.io/2020/08/09/groupnorm.html

#### Findings of normalization

#### Batch Norm Findings
** Toatal wrong predictions: 132
![image](https://user-images.githubusercontent.com/10822997/121735848-5291bf80-cb14-11eb-8639-936f03d459f2.png)

#### Layer Norm Findings
** Total wrong predictions- 71
![image](https://user-images.githubusercontent.com/10822997/121735958-7a812300-cb14-11eb-9f75-807ad880b9a8.png)

#### Group Norm Findings
** Total wrong predictions - 71
![image](https://user-images.githubusercontent.com/10822997/121736413-29256380-cb15-11eb-9230-25addc167828.png)


#### Training accuracy / Training loss  / Test accuracy / Test loss

![image](https://user-images.githubusercontent.com/10822997/121736336-0eeb8580-cb15-11eb-9f05-67de648f44ef.png)


### * Layer Norm and Group Norm performed similarly and has lower misclassified images than Batch Norm. 
### * Layer Norm and Group Norm has exhibited consistency in test loss and accuracy  but Batch Norm has ups and downs


