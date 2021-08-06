# Session 13 - ViT - An Image is worth 16x16 words



# Assignmet 

1. Let's review this blog on using ViT for Cats vs Dogs. Your assignment is to implement this blog and train the ViT model for Cats vs Dogs. If you wish you can use transfer learning.
  1. Share the link to the README that describes your CATS vs DOGS training using VIT. Expecting to see the training logs (at least 1) there.  
  2. Share the link to the notebook where I can find your Cats vs Dogs Training
  3. Expecting a Separate or same README to explain your understanding of all the Classes that we covered in the class. 
  
 
 
 1. Data Set Downloaded from Kaggle (https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)


        Train Data: 25000
        Test Data: 12500
   
  ### Random Samples of the data
  
  ![Screenshot 2021-08-06 at 10 08 32 PM](https://user-images.githubusercontent.com/73247157/128543427-47df246b-9a78-4759-9e42-aed4aed714b3.png)
  
 #### Train Validation Test Split of the data 
  
    Train Data: 20000
    Validation Data: 5000
    Test Data: 12500

 #### Effecient Attention
 
     efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
      )
      
    model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
    ).to(device)
    
 ###  Notebook Link - 
 
