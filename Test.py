import random
import cv2 as photo
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, img_as_float32
from sklearn.decomposition import KernelPCA
from sklearn import kernel_ridge
import os
from os import listdir
from progress.bar import Bar # Style — literally just style

"""
Run Blur and divide on a specified image
"""
def blurdivide(img, name):
    if not img.dtype == "float32":
        img = img_as_float32(img)
    sigma = ndimage.standard_deviation(img)
    numerator = filters.median(img)  # default is 3x3, same as RLE suggested
    denominator = filters.gaussian(img, sigma=3)
    ratio = numerator / denominator

    photo.imwrite("blur_divide_{name}.tif".format(name=name), ratio)
    photo.imwrite("blur_divide_Sobel{name}.tif".format(
        name=name), filters.sobel(ratio))


"""
Convert images to rbg
"""
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

"""
Style Points 
"""
def BarProg(TempNext,Temp):
    if(TempNext > Temp):
        return TempNext
    return 0
"""
Run Kernel PCA on images in a directory
"""
def K_PCA(kernel,transform,invTrans):
    eigen_solver = "dense"  # auto|dense|arpack
    n_jobs = -1  # -1 means all cores
   
    # get the path/directory
    print("collecting images")
    folder_dir = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess"
    i = 0
    numTifs=0
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            numTifs+=1
    TempBar = Bar('Fitting a {kernel} Kernel'.format(kernel = kernel),fill='█',index=0,max=numTifs)
    print("Starting Fit")
    kpca = KernelPCA(n_components=numTifs, kernel=kernel,eigen_solver=eigen_solver, n_jobs=n_jobs,fit_inverse_transform=True)
    for image in os.listdir(folder_dir):
        # check if the image ends with png
        if (image.endswith(".tif")):
        #    print("Fitting: {images}".format(images = image))
           img = photo.imread(image)
           img = rgb2gray(img)
           kpca.fit(img)
           i+=1
           TempBar.next(i) 
    TempBar.finish()
    if transform:
        plotHistogramTransforms(kernel,kpca,folder_dir,numTifs)
    if invTrans:
        plotHistogramInverseTransforms(kernel,kpca,folder_dir,numTifs)
    print("fit completed! Plotting Eigenvalues and saving Figure")
    plt.plot(kpca.eigenvalues_)
    plt.savefig("EigenValues_for_{kernel}_on_47".format(kernel = kernel))
    plt.clf()
    print('Done with {kernel}'.format(kernel = kernel))

"""
Plot the histograms of each KPCA transform
"""
def plotHistogramTransforms(kernel,kpca,folder_dir,numTifs):
    print('Plotting Histograms for Transformed Images using {kernel} kernel'.format(kernel = kernel))
    TempBar = Bar('Fitting a {kernel} Kernel'.format(kernel = kernel),fill='█',index=0,max=numTifs)
    i = 0
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            print("Transforming on Fit KPCA: {im}".format(im  = image))
            img = photo.imread(image)
            img = rgb2gray(img)
            out = kpca.inverse_transform(kpca.transform(img))
            out = img_as_float32(out)
            actualOut = out[:,1]
            Histogram1 = photo.calcHist([actualOut],[0],mask=None,histSize=[256],ranges=[0,256])
            plt.plot(Histogram1,'r')
            plt.savefig("Histogram_{image}_{kernel}_inv_transform_KPCA_47.png".format(image=image,kernel = kernel))
            plt.clf()
            i+=1
            TempBar.next(i)
    TempBar.finish()
"""
Plot the histograms of each of the KPCA inverse transform
"""    
def plotHistogramInverseTransforms(kernel,kpca,folder_dir,numTifs):
    print('Plotting Histograms for Inverse Transform Images using {kernel} kernel'.format(kernel = kernel))
    TempBar = Bar('Fitting a {kernel} Kernel'.format(kernel = kernel),fill='█',index=0,max=numTifs)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            print("Transforming on Fit KPCA: {im}".format(im  = image))
            img = photo.imread(image)
            img = rgb2gray(img)
            out = kpca.inverse_transform(kpca.transform(img))
            out = img_as_float32(out)
            actualOut = out[:,1]
            Histogram1 = photo.calcHist([actualOut],[0],mask=None,histSize=[256],ranges=[0,256])
            plt.plot(Histogram1,'r')
            plt.savefig("Histogram_{image}_{kernel}_inv_transform_KPCA_47.png".format(image=image,kernel = kernel))
            plt.clf()
            i+=1
            TempBar.next(i)
    TempBar.finish()

"""
Take the log of each image
"""
def Log(img,name):
    out = (np.log(img + 1)) 
    out = np.array(out, dtype = np.uint8)
    photo.imwrite('Log_Texture_{name}.tif'.format(name=name),out)
    return out

"""
Generate a pure histogram for a raw image
"""
def HistogramGen():
    folder_dir = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/"
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            print("Transforming on Fit KPCA: {im}".format(im  = image))
            img = photo.imread(image)
            img = rgb2gray(img)
            img = img_as_float32(img)
            actualOut = img[:,1]
            Histogram1 = photo.calcHist([actualOut],[0],mask=None,histSize=[256],ranges=[0,256])
            plt.plot(Histogram1,'r')
            plt.savefig("Histogram_{image}_RAW.png".format(image=image))
            plt.clf()


kernels = ['rbf','linear','cosine','sigmoid']
for k in kernels:
    K_PCA(k,False,False)

# HistogramGen()

# blurdivide(dim_2,"Band_07")
# blurdivide(dim_3,"Band_08")

# Min Ask - Max Bid = bid-ask spread

