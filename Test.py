## This is the script version of this KPCA implementation 
import random
import cv2 as photo
from scipy import linalg, ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, img_as_float32
from sklearn.decomposition import KernelPCA
from sklearn import kernel_ridge
import os
from os import listdir
from progress.bar import Bar # Style — literally just style
import csv



"""
Run Blur and divide on a specified image
"""
def blurdivide(img,name,save):
    if not img.dtype == "float32":
        img = img_as_float32(img)
    sigma = ndimage.standard_deviation(img)
    numerator = filters.median(img)  # default is 3x3, same as RLE suggested
    denominator = filters.gaussian(img, sigma=3)
    ratio = numerator / denominator
    if save:
        photo.imwrite("blur_divide_{name}.tif".format(name=name), ratio)
        photo.imwrite("blur_divide_Sobel{name}.tif".format(
            name=name), filters.sobel(ratio))
    return ratio


"""
Convert images to rbg
"""
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

"""
Style Points 
"""

"""
Run Kernel PCA on images in a directory
"""
def K_PCA(kernel,transform,invTrans,eigenvalues,fid,Contrast):
    eigen_solver = "dense"  # auto|dense|arpack
    n_jobs = -1  # -1 means all cores
    # get the path/directory
    print("collecting images")
    folder_dir = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess"
    i = 0
    numTifs=0
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif") and image != gt):
            numTifs+=1
    TempBar = Bar('Fitting a {kernel} Kernel'.format(kernel = kernel),fill='█',index=0,max=numTifs)
    kpca = KernelPCA(n_components=numTifs, kernel=kernel,eigen_solver=eigen_solver, n_jobs=n_jobs,fit_inverse_transform=True)
    for image in os.listdir(folder_dir):
        # check if the image ends with png
        if (image.endswith(".tif") and image != gt):
           img = photo.imread(image)
           img = rgb2gray(img)
           kpca.fit(img)
           i+=1
           TempBar.next() 
    TempBar.finish()
    if fid:
        Calculate_FID_From_Transformed_Images(kernel,kpca,folder_dir,numTifs)
    if Contrast:
        Calculate_Contrast_From_Inverse_Transformed_Images(kernel,kpca,folder_dir,numTifs)
    if transform:
        plotHistogramTransforms(kernel,kpca,folder_dir,numTifs)
    if invTrans:
        plotHistogramInverseTransforms(kernel,kpca,folder_dir,numTifs)
    if eigenvalues:
        print("fit completed! Plotting Eigenvalues and saving Figure")
        plt.plot(kpca.eigenvalues_)
        plt.savefig("EigenValues_for_{kernel}_on_47".format(kernel = kernel))
        plt.clf()
    print('Done with {kernel}'.format(kernel = kernel))

"""
Calculate Statistics accross 
"""
def Calculate_Statistics(_type,folder_dir,fid,Contrast,transform):
    numTifs = 0
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif") and image != gt):
            numTifs+=1
    if fid:
        Calculate_FID_Of_Images(_type,folder_dir,numTifs)
    if Contrast:
        Calculate_Contrast_Of_Images(_type,folder_dir,numTifs)
    if transform:
        plotHistogramTransforms(None,_type,folder_dir,numTifs)



"""
Calculates statistics for non-KPCA Transfroms 
"""
def Calculate_FID_Of_Images(type_of_transform,folder_dir,numTifs):
    TempBar = Bar('Calculating FID Scores for {kernel} type'.format(kernel = type_of_transform),fill='.',index=0,max=numTifs)
    fields = ["File", "FID Score"]
    file = open('FID_Scores_for_{kernel}.csv'.format(kernel = type_of_transform), 'w')
    writer = csv.writer(file)
    writer.writerow(fields)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            row_filler = []
            img = photo.imread(image,photo.IMREAD_ANYDEPTH)
            img = np.uint8(img)
            FID = fid(ground_truth,img)
            row_filler.append(str(image))
            row_filler.append(str(FID))
            writer.writerow(row_filler)
            row_filler.clear()
            TempBar.next()
    TempBar.finish()
    file.close()


"""
Calculates statistics for non-KPCA Transfroms 
"""
def Calculate_Contrast_Of_Images(type_of_transform,folder_dir,numTifs):
    TempBar = Bar('Transforming on a {kernel} Kernel'.format(kernel = type_of_transform),fill='.',index=0,max=numTifs)
    fields = ["File", "Contrast Score"]
    file = open('Contrast_Scores_for_{kernel}.csv'.format(kernel = type_of_transform), 'w')
    writer = csv.writer(file)
    writer.writerow(fields)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            row_filler = []
            img = photo.imread(image,photo.IMREAD_ANYDEPTH)
            contrast = np.std(img)
            row_filler.append(str(image))
            row_filler.append(str(contrast))
            writer.writerow(row_filler)
            row_filler.clear()
            TempBar.next()
    TempBar.finish()
    file.close()



def BlurAndDivideEnMass(folder_dir,numTifs):
    TempBar = Bar('Transforming on a {kernel} Kernel'.format(kernel = "Blur and Divide"),fill='.',index=0,max=numTifs)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            img = photo.imread(image)
            img = rgb2gray(img)
            img = blurdivide(img,image,True)
            TempBar.next()
    TempBar.finish()


"""
Plot the histograms of each KPCA transform on images
"""
def plotHistogramTransforms(kernel,kpca,folder_dir,numTifs):
    print('Plotting Histograms for Transformed Images using {kernel} kernel'.format(kernel = kernel),end=" ")
    TempBar = Bar('Transforming on a {kernel} Kernel'.format(kernel = kernel),fill='.',index=0,max=numTifs)
    i = 0
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif") and image != gt):
            img = photo.imread(image)
            img = rgb2gray(img)
            out = kpca.inverse_transform(kpca.transform(img))
            out = img_as_float32(out)
            actualOut = out[:,1]
            Histogram1 = photo.calcHist([actualOut],[0],mask=None,histSize=[256],ranges=[0,256])
            cumulative = Histogram1.cumsum()
            normalized = cumulative * Histogram1.max() / cumulative.max()
            plt.plot(normalized,'r')
            plt.savefig("Histogram_{image}_{kernel}_inv_transform_KPCA_47.png".format(image=image,kernel = kernel))
            plt.clf()
            i+=1
            TempBar.next()
    TempBar.finish()
    
"""
Plot the histograms of each of the KPCA inverse transform
"""    
def plotHistogramInverseTransforms(kernel,kpca,folder_dir,numTifs):
    print('Plotting Histograms for Inverse Transform Images using {kernel} kernel'.format(kernel = kernel),end=" ")
    TempBar = Bar('Transforming and inverting on a {kernel} Kernel'.format(kernel = kernel),fill='.',index=0,max=numTifs)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif") and image != gt):
            img = photo.imread(image)
            img = rgb2gray(img)
            out = kpca.inverse_transform(kpca.transform(img))
            out = img_as_float32(out)
            actualOut = out[:,1]
            cumulative = Histogram1.cumsum()
            normalized = cumulative * Histogram1.max() / cumulative.max()
            Histogram1 = photo.calcHist([actualOut],[0],mask=None,histSize=[256],ranges=[0,256])
            plt.plot(normalized,'r')
            plt.savefig("Histogram_{image}_{kernel}_inv_transform_KPCA_47.png".format(image=image,kernel = kernel))
            plt.clf()
            i+=1
            TempBar.next()
    TempBar.finish()

"""
Take the log of each image
"""
def Log(img,name):
    out = (np.log(img + 1)) 
    out = np.array(out, dtype = np.uint8)
    photo.imwrite('Log_Texture_{name}.tif'.format(name=name),out)
    return 
    out
"""
Cites: https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
"""
def fid(ground_truth, transformed_image):
     # calculate mean and covariance statistics
     mu1, sigma1 = ground_truth.mean(axis=0), np.cov(ground_truth, rowvar=False)
     mu2, sigma2 = transformed_image.mean(axis=0), np.cov(transformed_image,  rowvar=False)
     # calculate sum squared difference between means
     ssdiff = np.sum((mu1 - mu2)**2.0)
     # calculate sqrt of product between cov
     covmean = linalg.sqrtm(sigma1.dot(sigma2))
     # check and correct imaginary numbers from sqrt
     if np.iscomplexobj(covmean):
       covmean = covmean.real
     # calculate score
     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
     return fid

"""
From : https://arxiv.org/pdf/2106.13156.pdf
Calculates the Distance in disjointness between two images, the goal and the transformed 
And shows how closely an image matches the style of another image. 
"""
def Calculate_FID_From_Transformed_Images(kernel,kpca,folder_dir,numTifs):
    TempBar = Bar('Transforming on a {kernel} Kernel'.format(kernel = kernel),fill='.',index=0,max=numTifs)
    fields = ["File", "FID Score"]
    rows = []
    file = open('FID_Scores_for_{kernel}.csv'.format(kernel = kernel), 'w')
    writer = csv.writer(file)
    writer.writerow(fields)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            row_filler = []
            img = photo.imread(image)
            img = rgb2gray(img)
            out = kpca.inverse_transform(kpca.transform(img))
            out = img_as_float32(out)
            FID = fid(ground_truth,out)
            row_filler.append(str(image))
            row_filler.append(str(FID))
            writer.writerow(row_filler)
            row_filler.clear()
            TempBar.next()
    TempBar.finish()
    file.close()

"""
Calculate RMF Contrast
https://en.wikipedia.org/wiki/Contrast_(vision)#RMS_contrast
"""
def Calculate_Contrast_From_Inverse_Transformed_Images(kernel,kpca,folder_dir,numTifs):
    TempBar = Bar('Transforming on a {kernel} Kernel'.format(kernel = kernel),fill='.',index=0,max=numTifs)
    fields = ["File", "Contrast Score"]
    rows = []
    file = open('Contrast_Scores_for_{kernel}.csv'.format(kernel = kernel), 'w')
    writer = csv.writer(file)
    writer.writerow(fields)
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif")):
            row_filler = []
            img = photo.imread(image)
            img = rgb2gray(img)
            out = kpca.inverse_transform(kpca.transform(img))
            out = img_as_float32(out)
            contrast = np.std(out)
            row_filler.append(str(image))
            row_filler.append(str(contrast))
            writer.writerow(row_filler)
            row_filler.clear()
            TempBar.next()
    TempBar.finish()
    file.close()
    
"""
Generate a pure histogram for a raw image
"""
def HistogramGen():
    folder_dir = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/"
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif") and image != gt):
            img = photo.imread(image)
            img = rgb2gray(img)
            img = img_as_float32(img)
            actualOut = img[:,1]
            actualOut.flatten()
            Histogram1 = photo.calcHist([actualOut],[0],mask=None,histSize=[256],ranges=[0,256])
            cumulative = Histogram1.cumsum()
            cdf_normalized = cumulative * Histogram1.max() / cumulative.max()
            plt.plot(Histogram1,'r')
            plt.savefig("Histogram_{image}_RAW.png".format(image=image))
            plt.clf()

ground_truth = rgb2gray(photo.imread("/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/Ground_Truth_Image.tiff"))
gt = "/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/Ground_Truth_Image.png"
# Based on Eigenvalues, sigmoid was removed 



def Run_KPCA_Stuff():
    kernels = ['rbf','linear','cosine']
    for k in kernels:
        K_PCA(k,False,False,False,False,False)


def run_blur_divide():
    folder_dir = '/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/'
    numTifs = 0
    for image in os.listdir(folder_dir):
        if (image.endswith(".tif") and image != gt):
            numTifs+=1
    BlurAndDivideEnMass(folder_dir=folder_dir,numTifs=numTifs)
    
# run_blur_divide()

Calculate_Statistics(_type='PCA',folder_dir='/Users/chris/Documents/UR/SENIOR/FinalSemester/CSC249/Final/JubPalProcess/',fid=True,Contrast=True,transform=False)
