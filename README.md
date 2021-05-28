# PART 1

# Train a Simple Neural Network using Microsoft Excel

### Neural Network Diagram
<img width="539" alt="image" src="https://user-images.githubusercontent.com/73247157/120009954-33663e80-bffa-11eb-98b3-d7e235a1724a.png">


> 1. Draw the neural network diagram as shown in the figure.
> 2. Connect all the neurons using arrows and mark them with appropriate names and values.

---

### FeedForward Equations
![image](https://user-images.githubusercontent.com/73247157/120010011-46790e80-bffa-11eb-8809-5283b09cb3c9.png)

> 1. Write all the feedforward equations using the above diagram as reference.
> 2. Use sigmoid as the activation function for the hidden and output layers.

---

### Backpropagation Equations
![image](https://user-images.githubusercontent.com/73247157/120010087-5c86cf00-bffa-11eb-913b-bdfaba575aee.png)


### Training the Neural Network
![image](https://user-images.githubusercontent.com/73247157/120010232-863ff600-bffa-11eb-9e09-27d118ed0aee.png)


> 1. Initialize all the values of the neurons and weights as shown in the neural network diagram.
> 2. Write the equations of the weights and their gradients using the above equations and choosing the right cell numbers.
> 3. Use a constant learning rate to update the weights. This will be changed to observe the effect of learning rate on loss during training.
---

### Loss Graphs
<img width="736" alt="image" src="https://user-images.githubusercontent.com/73247157/120010299-9a83f300-bffa-11eb-9549-ea526a8c2e27.png">


> ### Observation: 
> We can observe that higher the value of learning rate, higher the rate of convergence of loss for this particular problem. This is not true in most deep neural network problems as the learning rate is generally kept low to update the weights slowly. 

---
# Part-2
On MNIST data, create a model to get 
99.4% validation accuracy
Less than 20k Parameters
You can use anything from above you want. 
Less than 20 Epochs
Have used BN, Dropout, a Fully connected layer, have used GAP. 

### Final Results
1. Achieved 99.4 test accuracy in 15th epoch
2. Total Parameters 9848
3. Droput = 0.04
4. Augmentation = Yes
5. Optimizer = SGD 
6. BatchNorm  = Yes
7. 1x1 Con Layer = Yes, 2 times
8. Max Channels = 20

### Model Architecture
![image](https://user-images.githubusercontent.com/73247157/120020888-f6a14400-c007-11eb-8956-0e000a6386b2.png)


2 3x3 conv layers followed by 1 1x1 con layer
![image](https://user-images.githubusercontent.com/73247157/120021044-2fd9b400-c008-11eb-85a8-f5e657359850.png)


### Model Results
  0%|          | 0/1875 [00:00<?, ?it/s]epoches: 1
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:84: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
Loss=0.1682090163230896 Batch_id=1874 Accuracy=91.70: 100%|██████████| 1875/1875 [00:25<00:00, 73.02it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0532, Accuracy: 9834/10000 (98.34%)

epoches: 2
Loss=0.04723441228270531 Batch_id=1874 Accuracy=97.19: 100%|██████████| 1875/1875 [00:25<00:00, 72.21it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0388, Accuracy: 9874/10000 (98.74%)

epoches: 3
Loss=0.03079293854534626 Batch_id=1874 Accuracy=97.60: 100%|██████████| 1875/1875 [00:25<00:00, 73.86it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0321, Accuracy: 9895/10000 (98.95%)

epoches: 4
Loss=0.003184626577422023 Batch_id=1874 Accuracy=97.89: 100%|██████████| 1875/1875 [00:25<00:00, 72.77it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0261, Accuracy: 9910/10000 (99.10%)

epoches: 5
Loss=0.024312244728207588 Batch_id=1874 Accuracy=98.11: 100%|██████████| 1875/1875 [00:26<00:00, 71.98it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0258, Accuracy: 9912/10000 (99.12%)

epoches: 6
Loss=0.0924961194396019 Batch_id=1874 Accuracy=98.25: 100%|██████████| 1875/1875 [00:25<00:00, 72.61it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0207, Accuracy: 9931/10000 (99.31%)

epoches: 7
Loss=0.05757199600338936 Batch_id=1874 Accuracy=98.31: 100%|██████████| 1875/1875 [00:26<00:00, 72.03it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0226, Accuracy: 9920/10000 (99.20%)

epoches: 8
Loss=0.24231401085853577 Batch_id=1874 Accuracy=98.37: 100%|██████████| 1875/1875 [00:25<00:00, 73.19it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0204, Accuracy: 9926/10000 (99.26%)

epoches: 9
Loss=0.00595014076679945 Batch_id=1874 Accuracy=98.44: 100%|██████████| 1875/1875 [00:26<00:00, 71.35it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0219, Accuracy: 9931/10000 (99.31%)

epoches: 10
Loss=0.005863219499588013 Batch_id=1874 Accuracy=98.48: 100%|██████████| 1875/1875 [00:25<00:00, 73.14it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0232, Accuracy: 9913/10000 (99.13%)

epoches: 11
Loss=0.0951457992196083 Batch_id=1874 Accuracy=98.59: 100%|██████████| 1875/1875 [00:26<00:00, 72.10it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0233, Accuracy: 9929/10000 (99.29%)

epoches: 12
Loss=0.0397578626871109 Batch_id=1874 Accuracy=98.59: 100%|██████████| 1875/1875 [00:26<00:00, 71.80it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0184, Accuracy: 9938/10000 (99.38%)

epoches: 13
Loss=0.010606668889522552 Batch_id=1874 Accuracy=98.69: 100%|██████████| 1875/1875 [00:25<00:00, 72.79it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0233, Accuracy: 9927/10000 (99.27%)

epoches: 14
Loss=0.02882736176252365 Batch_id=1874 Accuracy=98.66: 100%|██████████| 1875/1875 [00:26<00:00, 71.54it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0221, Accuracy: 9925/10000 (99.25%)

epoches: 15
Loss=0.01114226970821619 Batch_id=1874 Accuracy=98.72: 100%|██████████| 1875/1875 [00:26<00:00, 71.01it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0171, Accuracy: 9944/10000 (99.44%)

epoches: 16
Loss=0.017643120139837265 Batch_id=1874 Accuracy=98.64: 100%|██████████| 1875/1875 [00:26<00:00, 71.01it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0187, Accuracy: 9941/10000 (99.41%)

epoches: 17
Loss=0.002063074382022023 Batch_id=1874 Accuracy=98.72: 100%|██████████| 1875/1875 [00:25<00:00, 72.37it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0188, Accuracy: 9935/10000 (99.35%)

epoches: 18
Loss=0.012451915070414543 Batch_id=1874 Accuracy=98.75: 100%|██████████| 1875/1875 [00:26<00:00, 71.65it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9938/10000 (99.38%)

epoches: 19
Loss=0.006165508180856705 Batch_id=1874 Accuracy=98.80: 100%|██████████| 1875/1875 [00:26<00:00, 70.91it/s]

Test set: Average loss: 0.0213, Accuracy: 9930/10000 (99.30%)
