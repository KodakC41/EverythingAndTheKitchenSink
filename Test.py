# import jubpalprocess as proc
# import jubpalfunctions as func
import cv2 as photo
from scipy import ndimage
from skimage import img_as_float32
import imageio as iio
import sklearn.decomposition
import numpy as np

img = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/Cactus.jpeg"

def blurdivide(img,sigma):
	if not img.dtype == "float32":
		img = img_as_float32(img)
	# print("Creating a numerator as 3x3 median")
	numerator = photo.medianBlur(img,3) # default is 3x3, same as RLE suggested
	# print("Creating a denominator with radius/sigma = 50 Gaussian blur on denominator (RLE does 101x101 box blur)")
	denominator = photo.GaussianBlur(img,(3, 3),sigmaX=sigma)
	ratio = numerator / denominator
	return ratio
img = photo.imread(img)
sigma = ndimage.standard_deviation(img)
bd = blurdivide(img=img,sigma=sigma)
final_step =  bd / img
# photo.imwrite("blurdiv.png",final_step)



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
dim_2 = rgb2gray(img)

import skimage.io as io 
import numpy as np 
from sklearn.decomposition import KernelPCA 

def K_PCA(img):
    img = np.array(img, dtype='float32') 
    # Poly is good
    row = img.shape[0] 
    col = img.shape[1] 
    print(img.size,row,col)
    ras_shape = (row * col, 0) 
    pca = KernelPCA(n_components=759, 
                            kernel="poly", 
                            fit_inverse_transform=False, eigen_solver="dense",
                            gamma=2).fit_transform(img) 
    print(pca.size,pca.shape[0],pca.shape[1])
    new_shape = (pca.shape[0], pca.shape[1], -1) 
    kpca = pca.reshape(new_shape) 
    # saved_image = "kernel_pca.tif"
    io.imsave("Kernel_PCA.tif", kpca)
    return kpca
photo.imwrite("KPCA_over_Blur_Divide.png",final_step/K_PCA(dim_2))


