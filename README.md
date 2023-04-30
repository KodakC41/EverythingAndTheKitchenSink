# Recovering Palimpsest Data using Linear, Non-Linear, and Neural Network Approaches


## Methods Used

### In L3Harris Envi
- PCA
- ICA 
- MNF
- Texture Co-Occurence
- Spectral Angle Mapper (SAM)

### Included Python Code ```Test.py```
- KPCA
  - RBF Kernel
  - Cosine Kernel 
- Keith Knox Blur and Divide

### Included in Python Code ```Lazarus(LAZ)GAN.py```
- A GAN for the denoising of this data 

Model Specs: 

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 800, 800, 16)      448       
                                                                 
 batch_normalization (BatchN  (None, 800, 800, 16)     64        
 ormalization)                                                   
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 800, 800, 16)      0         
                                                                 
 dropout (Dropout)           (None, 800, 800, 16)      0         
                                                                 
 conv2d_1 (Conv2D)           (None, 400, 400, 64)      9280      
                                                                 
 batch_normalization_1 (Batc  (None, 400, 400, 64)     256       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 400, 400, 64)      0         
                                                                 
 dropout_1 (Dropout)         (None, 400, 400, 64)      0         
                                                                 
 conv2d_2 (Conv2D)           (None, 200, 200, 128)     73856     
                                                                 
 batch_normalization_2 (Batc  (None, 200, 200, 128)    512       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 200, 200, 128)     0         
                                                                 
 conv2d_transpose (Conv2DTra  (None, 400, 400, 128)    147584    
 nspose)                                                         
                                                                 
 leaky_re_lu_3 (LeakyReLU)   (None, 400, 400, 128)     0         
                                                                 
 dropout_2 (Dropout)         (None, 400, 400, 128)     0         
                                                                 
 conv2d_transpose_1 (Conv2DT  (None, 800, 800, 64)     73792     
 ranspose)                                                       
                                                                 
 leaky_re_lu_4 (LeakyReLU)   (None, 800, 800, 64)      0         
                                                                 
 conv2d_transpose_2 (Conv2DT  (None, 1600, 1600, 16)   9232      
 ranspose)                                                       
                                                                 
 leaky_re_lu_5 (LeakyReLU)   (None, 1600, 1600, 16)    0         
                                                                 
 conv2d_transpose_3 (Conv2DT  (None, 1600, 1600, 3)    435       
 ranspose)                                                       
                                                                 
 batch_normalization_3 (Batc  (None, 1600, 1600, 3)    12        
 hNormalization)                                                 
                                                                 
 dropout_3 (Dropout)         (None, 1600, 1600, 3)     0         
                                                                 
=================================================================
Total params: 315,471
Trainable params: 315,049
Non-trainable params: 422
_________________________________________________________________
```
