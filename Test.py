# import jubpalprocess as proc
# import jubpalfunctions as func
import cv2 as photo
from scipy import ndimage
from skimage import img_as_float32
import imageio as iio
import sklearn.decomposition
import numpy as np

img = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/Court_stylized_BubbleWrape.jpg"

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
# final_step =  bd / img
# photo.imwrite("blurdiv.png",final_step)



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
dim_2 = img

def K_PCA(img):
	img = np.array(img, dtype='float32') 
	row = img.shape[0] 
	col = img.shape[1] 
	bands = img.shape[2] 
	ras_shape = (row * col, bands)
	img_array = img[:, :, :bands].reshape(ras_shape) 
	kpca = sklearn.decomposition.KernelPCA(n_components=4, 
                            kernel="poly", 
                            fit_inverse_transform=False, 
                            gamma=0.5).fit_transform(img_array)
	print("success")
	new_shape = (4, col, row) 
	kpca = kpca.reshape(new_shape)
	photo.imwrite("KPCA_S_Court.png",kpca)
K_PCA(dim_2)


