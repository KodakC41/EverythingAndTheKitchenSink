# import jubpalprocess as proc
# import jubpalfunctions as func
import random
import cv2 as photo
from scipy import ndimage
from skimage import img_as_float32
import numpy as np
import skimage.io as io 
import numpy as np 
import skimage
from sklearn.decomposition import KernelPCA 
img = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/CACTUS.jpg"

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
final_step =  bd
photo.imwrite("blurdiv.png",bd)



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
dim_2 = rgb2gray(img)

# 100% Stack overflow
def sp_noise(image):
    row,col = image.shape
    mean = 3
    var = 0.25
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

# For images of greater than 2D you need to resize it to an incredible degree
def K_PCA(img):
    img = img_as_float32(img)
    photo.imwrite("test_image.png",img)
    row = img.shape[0] 
    col = img.shape[1] 
    gn_img = img_as_float32(sp_noise(img))

    photo.imwrite("Noisy_test.png",sp_noise(img))
    # Poly is good 
    kernel = "poly"
    gn_img_test = img_as_float32(sp_noise(img))
    photo.imwrite("median_filtered.png",photo.medianBlur(gn_img_test,3))

    k_pca = KernelPCA(n_components=100000000, kernel=kernel, fit_inverse_transform=True,eigen_solver="randomized")
    results = k_pca.fit_transform(img)
    reconstructed_k_pca = k_pca.inverse_transform(k_pca.fit_transform(photo.medianBlur(gn_img_test,3)))
    reconstructed_k_pca_no_median = k_pca.inverse_transform(k_pca.fit_transform(gn_img_test))
    
    out = reconstructed_k_pca_no_median.reshape(row,col)
    out2 = reconstructed_k_pca.reshape(row,col)
    
    photo.imwrite('reconstructed_no_median{kernel}.png'.format(kernel = kernel),out)
    photo.imwrite('reconstructed_median{kernel}.png'.format(kernel = kernel),out2)
    return results

k = K_PCA(dim_2)


