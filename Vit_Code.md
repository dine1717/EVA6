


The overall architecture can be described easily in five simple steps:
1. Split an input image into patches
2. Get linear embeddings (representations) from each patch referred to as Patch Embeddings
3. Add positional embeddings and a [CLS] token to each of the Patch Embeddings
4. there is more to the CLS token that we would cover today. Would request you to consider the definition of CLS token as shared in the last class as wrong, as we need to further decode it.
Pass through a Transformer Encoder and get the output values for each of the [CLS] tokens.
5. Pass the representations of [CLS] tokens through an MLP Head to get final class predictions. 


Patch Embeddings:

        class PatchEmbeddings(nn.Module):
            """
            Image to Patch Embedding.

            """

        def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
            super().__init__()
            image_size = to_2tuple(image_size)
            patch_size = to_2tuple(patch_size)
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = num_patches

            self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, pixel_values):
            batch_size, num_channels, height, width = pixel_values.shape
            # FIXME look at relaxing size constraints
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                )
            x = self.projection(pixel_values).flatten(2).transpose(1, 2)
            return x
            

   
   
 This class converts the image into patch embedding
 Here we have taken image size of 224, patch size of 16 , channels=3 and embded_dim=768.  image and patch size are checjed to see if they are iterables or not and    if not iterable they return tuples (224,224) and (16,16). Now it will calculates the number of patches  ( 224 // 16) * (224 // 16)  =196
        
 The input image is splited into N patches and convreted into 768 embedding vectors by learnable convlution 
   
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
  The output value get flattened and needs to transposed and reshaped 
  
  ![image](https://user-images.githubusercontent.com/73247157/128554128-054f0106-7a6e-4286-91c4-f93dd48ec958.png)

  
  
  
  Construct the CLS token, postion and patch embeddings
  
    class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
  
  
To make patches position-aware, learnable 'position embedding' vectors are added to the patch embedding vectors. The position embedding vectors learn distance within the image thus neighboring ones have high similarity.

A learnable class token is prepended to the patch embedding vectors as the 0th vector.
197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors.

        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
  
  ![image](https://user-images.githubusercontent.com/73247157/128554169-a505a231-feb2-465e-9817-63417457522c.png)


VitConfig

        class ViTConfig():
          def __init__(
                self,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                is_encoder_decoder=False,
                image_size=224,
                patch_size=16,
                num_channels=3,
                **kwargs
            ):

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        
    
