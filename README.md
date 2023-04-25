# Recovering Palimpsest Data using Linear, Non-Linear, and Neural Network Approaches


## Methods Used

### In L3Harris Envi
- PCA
- ICA 
- MNF

### Included Python Code ```Test.py```
- KPCA
  - RBF Kernel
  - Cosine Kernel 
- Keith Knox Blur and Divide

### Included in Python Code ```GANarus.py```
- A GAN for the denoising of this data 

Model Specs: 

```python

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 1000, 1000, 16)    448       
                                                                 
 batch_normalization (BatchN  (None, 1000, 1000, 16)   64        
 ormalization)                                                   
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 1000, 1000, 16)    0         
                                                                 
 conv2d_1 (Conv2D)           (None, 500, 500, 64)      9280      
                                                                 
 batch_normalization_1 (Batc  (None, 500, 500, 64)     256       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 500, 500, 64)      0         
                                                                 
 conv2d_2 (Conv2D)           (None, 250, 250, 128)     73856     
                                                                 
 batch_normalization_2 (Batc  (None, 250, 250, 128)    512       
 hNormalization)                                                 
                                                                 
 conv2d_transpose (Conv2DTra  (None, 500, 500, 128)    147584    
 nspose)                                                         
                                                                 
 batch_normalization_3 (Batc  (None, 500, 500, 128)    512       
 hNormalization)                                                 
                                                                 
 p_re_lu (PReLU)             (None, 500, 500, 128)     32000000  
                                                                 
 conv2d_transpose_1 (Conv2DT  (None, 1000, 1000, 64)   73792     
 ranspose)                                                       
                                                                 
 batch_normalization_4 (Batc  (None, 1000, 1000, 64)   256       
 hNormalization)                                                 
                                                                 
 p_re_lu_1 (PReLU)           (None, 1000, 1000, 64)    64000000  
                                                                 
 conv2d_transpose_2 (Conv2DT  (None, 2000, 2000, 16)   9232      
 ranspose)                                                       
                                                                 
 batch_normalization_5 (Batc  (None, 2000, 2000, 16)   64        
 hNormalization)                                                 
                                                                 
 p_re_lu_2 (PReLU)           (None, 2000, 2000, 16)    64000000  
                                                                 
 conv2d_transpose_3 (Conv2DT  (None, 2000, 2000, 3)    435       
 ranspose)                                                       
                                                                 
 dropout (Dropout)           (None, 2000, 2000, 3)     0         
                                                                 
=================================================================
Total params: 160,316,291
Trainable params: 160,315,459
Non-trainable params: 832
_________________________________________________________________

```
