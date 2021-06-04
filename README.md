# **EVA6 Session5**

# Assignment Target 

1. Acheive 99.4%  accuraacy on MNIST datastet(this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 epochs
3. Less than 10000 Parameters


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
 1. Model is working but we see for some of the epochs its overfitting 

___________

# **Step 2**


 
 [Code 2](https://github.com/dine1717/EVA6/blob/Session5/Step_2.ipynb)
 
 # Target

 1. Reduce Model Parameters
 2. Use Batch Normalization to improve accuracy
 3. Use Gap to improve accuracy
 
# Result:
 
 1. Total Parameters: 8872
 2. Best Training Accuarcy: 99.45
 3. Best Test Accuarcy: 99.32
 
# Analysis:
 1. Model is performing good and in some epochs it is close to the target.

___________

# **Step 3**



 
 [Code 3](https://github.com/dine1717/EVA6/blob/Session5/Step_3.ipynb)
 
# Target
1. Increase aacuracy  by using dropout and Augmentation
 
# Result:
 
 1. Total Parameters: 8872
 2. Best Training Accuarcy: 99.14
 3. Best Test Accuarcy: 99.28
 
# Analysis:
 1. The model is under fitting and we can see some consisnet accuary in the test data  aroun 99.2 but our target is 99.4.
 

___________

# **Step 4**



 
 [Code 4](https://github.com/dine1717/EVA6/blob/Session5/Step_4.ipynb)
 
# Target
1. Apply Learning Rate Scheduler

 
# Result:
 
 1. Total Parameters: 8872
 2. Best Training Accuarcy: 99.05
 3. Best Test Accuarcy: 99.3
 
# Analysis:
 1. Model is performing good and in some epochs it is close to the target.
 2. Compared to previous iteration we increase our test accuarcy a bit but we stii need to reach the target.
 
 


___________

# **Step 5**



 
 [Code 5](https://github.com/dine1717/EVA6/blob/Session5/Step_5.ipynb)
 
# Target

1. Less than 7000 parameters
2. Less than 15 epochs
3. Test with Cyclic LR
4. Add small dropout of 5%

# Results

1. Number of Parameters = 6202 + 140 non trainable BN params
2. Best Train Accuracy = 98.97
3.Best Test Accuracy = 99.45

# Analysis

1.We pushed the model to achieve target with approx 6000 Parameters
2. Drop in accuracy was predicted because of making the training more difficult. But still we met the target.


