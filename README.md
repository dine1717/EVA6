
# EVA6 Assignment-5
## Task
* Acheive 99.4% accuracy on MNIST dataset(this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
* Explain your 3 steps using these target, results, and analysis with links to your GitHub files

# **Step 1**
______________

[Code 1](https://github.com/dine1717/EVA6/blob/Session5/Step_1.ipynb)


# Target

 1. Settle on Architeture
 2. Visualize Data
 3. Define Data Loaders, Data Transformations and Image Normalization
 4. Design vanila model architeture.
 
# Result:
 
 1. Total Parameters: 147,344
 2. Best Training Accuarcy: 99.78
 3. Best Test Accuarcy: 99.11
 
# Analysis:
 1. Model is working but we see for some of the epochs its overfitting.
 2. Model will never reach 99.4 on test data as the train data already acheived a max of 99.78
 3. To avoid overfitting we need three things; 1.Drop0ut, 2. Fewer Parameters, 3. Batch normalization

___________

# **Step 2**


 
 [Code 2](https://github.com/dine1717/EVA6/blob/Session5/Step_2.ipynb)
 
 # Target

 1. Reduce Model Parameters
 2. Use Batch Normalization to improve accuracy
 3. Use Gap to improve accuracy
 
# Result:
 
 1. Total Parameters: 10616
 2. Best Training Accuarcy: 99.52
 3. Best Test Accuarcy: 99.37
 
# Analysis:
 1. Model is performing good and in some epochs it is close to the target.
 2. MNIST doesn't need 145k parameters even 10k is good enough
 3. Model is not overfitting, so we can reach 99.4 accuracy on test
 4. Data augmentation might help

___________

# **Step 3**



 
 [Code 3](https://github.com/dine1717/EVA6/blob/Session5/Step_3.ipynb)
 
# Target
1. Increase aacuracy  by using dropout and Augmentation
 
# Result:
 
 1. Total Parameters: 10616
 2. Best Training Accuarcy: 99.28
 3. Best Test Accuarcy: 99.45
 
# Analysis:
 1. The model attained 99.4 in 8th epoch but it is not consistent 
 2. Try going deeper by adding padding in the initial layers
 3. Add 1x1 conv block after GAP
 

___________

# **Step 4**



 
 [Code 4](https://github.com/dine1717/EVA6/blob/Session5/Step_5.ipynb)
 
# Target
1. Apply 1x1 after GAP
2. Add more layers 

 
# Result:
 
 1. Total Parameters: 8172
 2. Best Training Accuarcy: 99.32
 3. Best Test Accuarcy: 99.5
 
# Analysis:
 1. Model is performing attained 99.39 for the first time in the 10th epoch and max acciracy is 99.50
 2. 8k parameters are good enough
 3. Going deeper with 7 conv layers helped
 4. Added 1x1 conv layer after GAP
 5. Consistently above 99.35 in the last 5 epochs
 
### Logs
 
EPOCH: 10
Loss=0.05837603285908699 Batch_id=468 Accuracy=98.59: 100%|██████████| 469/469 [00:21<00:00, 22.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0212, Accuracy: 9939/10000 (99.39%)

EPOCH: 11
Loss=0.06231403350830078 Batch_id=468 Accuracy=98.69: 100%|██████████| 469/469 [00:21<00:00, 22.28it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0234, Accuracy: 9932/10000 (99.32%)

EPOCH: 12
Loss=0.029857659712433815 Batch_id=468 Accuracy=98.66: 100%|██████████| 469/469 [00:21<00:00, 22.32it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0227, Accuracy: 9937/10000 (99.37%)

EPOCH: 13
Loss=0.03955037519335747 Batch_id=468 Accuracy=98.70: 100%|██████████| 469/469 [00:21<00:00, 22.28it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0182, Accuracy: 9951/10000 (99.51%)

EPOCH: 14
Loss=0.0427081398665905 Batch_id=468 Accuracy=98.74: 100%|██████████| 469/469 [00:21<00:00, 22.33it/s]
Test set: Average loss: 0.0192, Accuracy: 9940/10000 (99.40%)

___________

# **Step 5**



 
 [Code 5](https://github.com/dine1717/EVA6/blob/Session5/Step_5.ipynb)
 
# Target

1. Less than 9000 parameters
2. Less than 15 epochs
3. Test with Cyclic LR
4. Add small dropout of 5%

# Results

1. Number of Parameters = 8172
2. Best Train Accuracy = 98.97
3.Best Test Accuracy = 99.46

# Analysis

1.We pushed the model to achieve target with approx 8k Parameters
2. Model consistently has 99.4 accuracy in the last 5 epocs
3. Onecycle LR is pure magic


### Logs

Loss=0.04271348938345909 Batch_id=468 Accuracy=98.72: 100%|██████████| 469/469 [00:20<00:00, 22.97it/s]Epoch: 9 LR: [0.046231902768540376]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0195, Accuracy: 9947/10000 (99.47%)

EPOCH: 10
Loss=0.07905734330415726 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:21<00:00, 22.15it/s]Epoch: 10 LR: [0.031703533067975895]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0212, Accuracy: 9939/10000 (99.39%)

EPOCH: 11
Loss=0.025937607511878014 Batch_id=468 Accuracy=98.85: 100%|██████████| 469/469 [00:21<00:00, 21.85it/s]Epoch: 11 LR: [0.018800902517922092]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9939/10000 (99.39%)

EPOCH: 12
Loss=0.08558371663093567 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:21<00:00, 22.29it/s]Epoch: 12 LR: [0.008670466465012771]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0168, Accuracy: 9946/10000 (99.46%)

EPOCH: 13
Loss=0.0028163206297904253 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:20<00:00, 22.78it/s]Epoch: 13 LR: [0.0022123586092353013]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0166, Accuracy: 9946/10000 (99.46%)

EPOCH: 14
Loss=0.006995941046625376 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:20<00:00, 22.75it/s]Epoch: 14 LR: [4.101745150496986e-07]


Test set: Average loss: 0.0161, Accuracy: 9944/10000 (99.44%)


# **Step 6**

##  *** Bonus ***

 
 [Code 6](hhttps://github.com/dine1717/EVA6/blob/Session5/Step_6.ipynb)
 
# Target

1. Less than 7000 parameters
2. Less than 15 epochs
3. Test with Cyclic LR
4. Add small dropout of 5%

# Results

1. Number of Parameters = 6750
2. Best Train Accuracy = 98.97
3.Best Test Accuracy = 99.46

# Analysis

1.We pushed the model to achieve target with approx 6.7k Parameters
2. Model consistently has 99.4 accuracy in the last 6 epocs except one in between
3. 10 is the max output channels we have used except last conv block


### Logs
Loss=0.10133526474237442 Batch_id=468 Accuracy=98.44: 100%|██████████| 469/469 [00:14<00:00, 33.07it/s]Epoch: 8 LR: [0.061095102215020056]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0233, Accuracy: 9938/10000 (99.38%)

EPOCH: 9
Loss=0.02401966042816639 Batch_id=468 Accuracy=98.53: 100%|██████████| 469/469 [00:14<00:00, 33.11it/s]Epoch: 9 LR: [0.046231902768540376]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0213, Accuracy: 9936/10000 (99.36%)

EPOCH: 10
Loss=0.11673140525817871 Batch_id=468 Accuracy=98.63: 100%|██████████| 469/469 [00:14<00:00, 32.94it/s]Epoch: 10 LR: [0.031703533067975895]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9945/10000 (99.45%)

EPOCH: 11
Loss=0.07196126878261566 Batch_id=468 Accuracy=98.71: 100%|██████████| 469/469 [00:14<00:00, 32.79it/s]Epoch: 11 LR: [0.018800902517922092]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0187, Accuracy: 9941/10000 (99.41%)

EPOCH: 12
Loss=0.14405079185962677 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:14<00:00, 33.21it/s]Epoch: 12 LR: [0.008670466465012771]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9938/10000 (99.38%)

EPOCH: 13
Loss=0.0874735489487648 Batch_id=468 Accuracy=98.88: 100%|██████████| 469/469 [00:14<00:00, 32.97it/s]Epoch: 13 LR: [0.0022123586092353013]

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0181, Accuracy: 9942/10000 (99.42%)

EPOCH: 14
Loss=0.010102950036525726 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:14<00:00, 32.71it/s]Epoch: 14 LR: [4.101745150496986e-07]


Test set: Average loss: 0.0174, Accuracy: 9943/10000 (99.43%)

![image](https://user-images.githubusercontent.com/10822997/120845251-2b744480-c58e-11eb-8cad-cfb864124772.png)
