Spatial Transformer

Team Members: Bharath,Sabeesh,Manu,Dinesh


Github link: https://github.com/dine1717/EVA6/blob/Session12/Session12_Spatial_Transformer.ipynb

colab link : https://github.com/dine1717/EVA6/blob/Session12/Session12_Spatial_Transformer.ipynb



A Spatial Transformer is an image model block that explicitly allows the spatial manipulation of data within a convolutional neural network. It gives CNNs the ability to actively spatially transform feature maps, conditional on the feature map itself, without any extra training supervision or modification to the optimisation process. Unlike pooling layers, where the receptive fields are fixed and local, the spatial transformer module is a dynamic mechanism that can actively spatially transform an image (or a feature map) by producing an appropriate transformation for each input sample. The transformation is then performed on the entire feature map (non-locally) and can include scaling, cropping, rotations, as well as non-rigid deformations.

![image](https://user-images.githubusercontent.com/73247157/127658957-c9d5e050-22c4-4604-9981-4ad660d2eac8.png)


Spatial transformer networks boils down to three main components :

1. The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
2. The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
3. The sampler uses the parameters of the transformation and applies it to the input image.







Training and Validation logs:

    Train Epoch: 1 [0/50000 (0%)]	Loss: 2.340884
    Train Epoch: 1 [32000/50000 (64%)]	Loss: 2.094854
    Test set: Average loss: 1.8471, Accuracy: 3473/10000 (35%)

    Train Epoch: 2 [0/50000 (0%)]	Loss: 2.044745
    Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.699151

    Test set: Average loss: 1.6293, Accuracy: 4031/10000 (40%)

    Train Epoch: 3 [0/50000 (0%)]	Loss: 1.883966
    Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.675035

    Test set: Average loss: 1.5102, Accuracy: 4632/10000 (46%)

    Train Epoch: 4 [0/50000 (0%)]	Loss: 1.626356
    Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.663602

    Test set: Average loss: 1.4342, Accuracy: 4907/10000 (49%)

    Train Epoch: 5 [0/50000 (0%)]	Loss: 1.532872
    Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.456992

    Test set: Average loss: 1.3778, Accuracy: 5061/10000 (51%)

    Train Epoch: 6 [0/50000 (0%)]	Loss: 1.436933
    Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.525159

    Test set: Average loss: 1.3373, Accuracy: 5302/10000 (53%)

    Train Epoch: 7 [0/50000 (0%)]	Loss: 1.227678
    Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.514829

    Test set: Average loss: 1.4663, Accuracy: 4828/10000 (48%)

    Train Epoch: 8 [0/50000 (0%)]	Loss: 1.805382
    Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.308874

    Test set: Average loss: 1.2816, Accuracy: 5505/10000 (55%)

    Train Epoch: 9 [0/50000 (0%)]	Loss: 1.172609
    Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.169108

    Test set: Average loss: 1.2384, Accuracy: 5674/10000 (57%)

    Train Epoch: 10 [0/50000 (0%)]	Loss: 1.354637
    Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.315861

    Test set: Average loss: 1.3187, Accuracy: 5442/10000 (54%)

    Train Epoch: 11 [0/50000 (0%)]	Loss: 1.275378
    Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.274624

    Test set: Average loss: 1.1975, Accuracy: 5915/10000 (59%)

    Train Epoch: 12 [0/50000 (0%)]	Loss: 1.275416
    Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.344771

    Test set: Average loss: 1.2241, Accuracy: 5728/10000 (57%)

    Train Epoch: 13 [0/50000 (0%)]	Loss: 1.354707
    Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.299348

    Test set: Average loss: 1.1612, Accuracy: 5991/10000 (60%)

    Train Epoch: 14 [0/50000 (0%)]	Loss: 1.073826
    Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.143067

    Test set: Average loss: 1.1332, Accuracy: 6109/10000 (61%)

    Train Epoch: 15 [0/50000 (0%)]	Loss: 1.230965
    Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.058437

    Test set: Average loss: 1.2236, Accuracy: 5731/10000 (57%)

    Train Epoch: 16 [0/50000 (0%)]	Loss: 1.268024
    Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.064889

    Test set: Average loss: 1.1530, Accuracy: 6021/10000 (60%)

    Train Epoch: 17 [0/50000 (0%)]	Loss: 1.162964
    Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.259275

    Test set: Average loss: 1.1335, Accuracy: 6097/10000 (61%)

    Train Epoch: 18 [0/50000 (0%)]	Loss: 0.963113
    Train Epoch: 18 [32000/50000 (64%)]	Loss: 0.970027

    Test set: Average loss: 1.1217, Accuracy: 6143/10000 (61%)

    Train Epoch: 19 [0/50000 (0%)]	Loss: 1.230440
    Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.244240

    Test set: Average loss: 1.1107, Accuracy: 6168/10000 (62%)

    Train Epoch: 20 [0/50000 (0%)]	Loss: 1.113915
    Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.183479

    Test set: Average loss: 1.0889, Accuracy: 6220/10000 (62%)

    Train Epoch: 21 [0/50000 (0%)]	Loss: 1.323742
    Train Epoch: 21 [32000/50000 (64%)]	Loss: 1.242497

    Test set: Average loss: 1.0750, Accuracy: 6339/10000 (63%)

    Train Epoch: 22 [0/50000 (0%)]	Loss: 1.230038
    Train Epoch: 22 [32000/50000 (64%)]	Loss: 0.852995

    Test set: Average loss: 1.1182, Accuracy: 6134/10000 (61%)

    Train Epoch: 23 [0/50000 (0%)]	Loss: 0.987514
    Train Epoch: 23 [32000/50000 (64%)]	Loss: 1.067162

    Test set: Average loss: 1.0345, Accuracy: 6467/10000 (65%)

    Train Epoch: 24 [0/50000 (0%)]	Loss: 0.935367
    Train Epoch: 24 [32000/50000 (64%)]	Loss: 1.065407

    Test set: Average loss: 1.0462, Accuracy: 6437/10000 (64%)

    Train Epoch: 25 [0/50000 (0%)]	Loss: 0.911777
    Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.009808

    Test set: Average loss: 1.1214, Accuracy: 6105/10000 (61%)

    Train Epoch: 26 [0/50000 (0%)]	Loss: 1.040082
    Train Epoch: 26 [32000/50000 (64%)]	Loss: 0.887740

    Test set: Average loss: 1.0556, Accuracy: 6340/10000 (63%)

    Train Epoch: 27 [0/50000 (0%)]	Loss: 0.990285
    Train Epoch: 27 [32000/50000 (64%)]	Loss: 0.908881

    Test set: Average loss: 1.0321, Accuracy: 6466/10000 (65%)

    Train Epoch: 28 [0/50000 (0%)]	Loss: 0.861332
    Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.895714

    Test set: Average loss: 1.0092, Accuracy: 6539/10000 (65%)

    Train Epoch: 29 [0/50000 (0%)]	Loss: 0.929652
    Train Epoch: 29 [32000/50000 (64%)]	Loss: 0.914139

    Test set: Average loss: 1.2248, Accuracy: 5833/10000 (58%)

    Train Epoch: 30 [0/50000 (0%)]	Loss: 1.220526
    Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.963922

    Test set: Average loss: 1.0790, Accuracy: 6354/10000 (64%)

    Train Epoch: 31 [0/50000 (0%)]	Loss: 0.878361
    Train Epoch: 31 [32000/50000 (64%)]	Loss: 0.883062

    Test set: Average loss: 1.0229, Accuracy: 6531/10000 (65%)

    Train Epoch: 32 [0/50000 (0%)]	Loss: 0.954260
    Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.869309

    Test set: Average loss: 1.0354, Accuracy: 6508/10000 (65%)

    Train Epoch: 33 [0/50000 (0%)]	Loss: 1.020282
    Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.678881

    Test set: Average loss: 1.0198, Accuracy: 6497/10000 (65%)

    Train Epoch: 34 [0/50000 (0%)]	Loss: 0.660955
    Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.726761

    Test set: Average loss: 1.2910, Accuracy: 5626/10000 (56%)

    Train Epoch: 35 [0/50000 (0%)]	Loss: 1.178028
    Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.766355

    Test set: Average loss: 1.0549, Accuracy: 6363/10000 (64%)

    Train Epoch: 36 [0/50000 (0%)]	Loss: 0.871979
    Train Epoch: 36 [32000/50000 (64%)]	Loss: 0.837307

    Test set: Average loss: 1.0024, Accuracy: 6577/10000 (66%)

    Train Epoch: 37 [0/50000 (0%)]	Loss: 0.917059
    Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.609563

    Test set: Average loss: 0.9995, Accuracy: 6600/10000 (66%)

    Train Epoch: 38 [0/50000 (0%)]	Loss: 0.846795
    Train Epoch: 38 [32000/50000 (64%)]	Loss: 0.663952

    Test set: Average loss: 1.0828, Accuracy: 6337/10000 (63%)

    Train Epoch: 39 [0/50000 (0%)]	Loss: 1.159349
    Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.652166

    Test set: Average loss: 1.0827, Accuracy: 6363/10000 (64%)

    Train Epoch: 40 [0/50000 (0%)]	Loss: 0.810317
    Train Epoch: 40 [32000/50000 (64%)]	Loss: 0.740000

    Test set: Average loss: 1.0168, Accuracy: 6561/10000 (66%)

    Train Epoch: 41 [0/50000 (0%)]	Loss: 0.824200
    Train Epoch: 41 [32000/50000 (64%)]	Loss: 0.660295

    Test set: Average loss: 1.0291, Accuracy: 6447/10000 (64%)

    Train Epoch: 42 [0/50000 (0%)]	Loss: 0.947073
    Train Epoch: 42 [32000/50000 (64%)]	Loss: 0.874291

    Test set: Average loss: 1.1453, Accuracy: 6180/10000 (62%)

    Train Epoch: 43 [0/50000 (0%)]	Loss: 1.020692
    Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.846220

    Test set: Average loss: 1.0519, Accuracy: 6513/10000 (65%)

    Train Epoch: 44 [0/50000 (0%)]	Loss: 0.907695
    Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.561350

    Test set: Average loss: 1.0560, Accuracy: 6453/10000 (65%)

    Train Epoch: 45 [0/50000 (0%)]	Loss: 0.495847
    Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.702567

    Test set: Average loss: 1.0929, Accuracy: 6352/10000 (64%)

    Train Epoch: 46 [0/50000 (0%)]	Loss: 0.667552
    Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.500723

    Test set: Average loss: 1.0594, Accuracy: 6459/10000 (65%)

    Train Epoch: 47 [0/50000 (0%)]	Loss: 0.863764
    Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.655614

    Test set: Average loss: 1.0307, Accuracy: 6551/10000 (66%)

    Train Epoch: 48 [0/50000 (0%)]	Loss: 0.692034
    Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.774511

    Test set: Average loss: 1.0196, Accuracy: 6606/10000 (66%)

    Train Epoch: 49 [0/50000 (0%)]	Loss: 0.634608
    Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.664859

    Test set: Average loss: 1.0208, Accuracy: 6588/10000 (66%)

    Train Epoch: 50 [0/50000 (0%)]	Loss: 0.660755
    Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.605677

    Test set: Average loss: 1.0031, Accuracy: 6664/10000 (67%)
