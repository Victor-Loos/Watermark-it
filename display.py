from matplotlib import pyplot as plt
import cv2

from watermark import convertImage

import numpy as np
from PIL import Image

import imageio.v3 as iio
import skimage.color
import skimage.util


# Read the image and convert them in the same way as the watermarking process
def imageProcessing(image, marque, Iresult, Mresult, x):
    # Get the size of the mark
    imageArray, colors, size = convertImage(Iresult)
    Msize = (int(((size[1]/2)/8)/x), int(((size[0]/2)/8)/x)) # DCT + Block + random

    # Compare width and height of Msize and multiply the smallest
    if Msize[0] < Msize[1]:
        nbCoords = Msize[0] * Msize[0]
    else:
        nbCoords = Msize[1] * Msize[1]
    Wlength = int(np.sqrt(nbCoords))

    # Image original :   
    readImage = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    # Image used for cover :
    _, usedImage, _ = convertImage(image)
    usedImage = Image.merge('YCbCr', usedImage).convert('RGB')
    usedImage = np.array(usedImage)

    # Image used for watermark :
    img = cv2.resize(cv2.imread(marque, cv2.IMREAD_GRAYSCALE), (Wlength, Wlength))
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
    markArray = np.array(img, dtype=float).reshape((Wlength, Wlength))
    marqueImg = Image.fromarray(markArray)

    # Result watermarked image :
    waterimg = cv2.imread(Iresult)
    waterimg = cv2.cvtColor(waterimg, cv2.COLOR_BGR2RGB)

    # Result recovered watermark :
    recovwater = cv2.imread(Mresult, cv2.IMREAD_GRAYSCALE)
    
    return readImage, usedImage, marqueImg, waterimg, recovwater


def imageProcessingText(image, Iresult):
    # Image used for cover :
    _, usedImage, _ = convertImage(image)
    usedImage = Image.merge('YCbCr', usedImage).convert('RGB')
    usedImage = np.array(usedImage)

    # Result watermarked image :
    waterimg = cv2.imread(Iresult)
    waterimg = cv2.cvtColor(waterimg, cv2.COLOR_BGR2RGB)
    
    return usedImage, waterimg


### Visualisation ###


def plot_table(data, marque):
    title_text = 'Text attack results'
    #fig_background_color = 'skyblue'
    #fig_border = 'steelblue'
    # Define the original text
    original_text = "Text : " + marque

    # Pop the headers from the data array
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]

    # Table data needs to be non-numeric text.
    cell_text = []
    for row in data:
        cell_text.append([str(x) for x in row])

    # Get some lists of color specs for row and column headers
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))

    # Define the width of each column
    col_widths = [0.9, 0.07]

    # Create the figure.
    plt.figure(linewidth=2,
            #edgecolor=fig_border,
            #facecolor=fig_background_color,
            tight_layout={'pad':1},
            figsize=(12, 6)
            )

    # Add a text box under the title
    plt.figtext(0.5, 0.9, original_text, ha='center', va='center', fontsize=12)#, fontweight='bold')

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        rowLoc='left',  # move row header to the left
                        cellLoc='left',  # move cell contents to the left
                        colLabels=column_headers,
                        colWidths=col_widths,
                        loc='center')

    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 2)
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)
    # Add title
    plt.suptitle(title_text)
    # Force the figure to update, so backends center objects correctly within the figure.
    plt.show()



#### Display results ####
def plotResult(readImage, marqueImg, waterimg, recovwater, x):
    readImage, usedImage, marqueImg, waterimg, recovwater  = imageProcessing(readImage, marqueImg, waterimg, recovwater, x)
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(ncols=4, figsize=(15, 3))
    # Adjust the height of the subplots
    fig.subplots_adjust(top=0.8, bottom=0.2)

    # Remove the ticks from the plots
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])

    # Plot your images in each subplot
    axes[0].set_title("Original cover image")
    axes[0].imshow(cv2.cvtColor(readImage, 1,))
    axes[0].set_xlabel(f'Size: {readImage.shape}')

    axes[1].set_title("Original mark image")
    axes[1].imshow(marqueImg)
    axes[1].set_xlabel(f'Size: {marqueImg.size}')

    axes[2].set_title("WaterMarked image")
    axes[2].imshow(cv2.cvtColor(waterimg, 1))
    axes[2].set_xlabel(f'Size: {waterimg.shape}')

    axes[3].set_title("Recovered mark image")
    axes[3].imshow(recovwater, cmap = 'gray')
    axes[3].set_xlabel(f'Size: {recovwater.shape}')

    plt.show()



####### Compare difference between original and watermarked image #######
def plotDiff(readImage, marqueImg, waterimg, recovwater, x):
    readImage, usedImage, marqueImg, waterimg, recovwater = imageProcessing(readImage, marqueImg, waterimg, recovwater, x)

    # Convert marqueImg from binary PIL to numpy array
    marqueNp = np.array(marqueImg)
    # Convert True to 1 and False to 0
    marqueNp = np.where(marqueNp == False, 255, 0)
    marqueNp = np.array(marqueNp, dtype = np.uint8)

    # Compute and Plot difference between original and watermarked image
    diffimg = cv2.absdiff(usedImage, waterimg)
    diffmark = cv2.absdiff(marqueNp, recovwater)

    # Number of different pixels
    pixelimg = np.where(diffimg > 0, 1, 0)
    pixelmark = np.where(diffmark > 0, 1, 0)


    #print("- Image :\nMean of difference :", str(np.mean(diffimg)), "\nMax of difference  : " + str(np.max(diffimg)), 
    #      "\nNumber of different pixels :", str(np.size(diffimg)-np.sum(pixelimg)) + " --> ", str(np.sum(pixelimg)), "/", str(np.size(diffimg)), "=", str(np.sum(pixelimg)/np.size(diffimg)*100)[:5], "% unchanged")
    Istats = "Mean of difference : " + str(np.mean(diffimg))[:6] + "\nMax of difference : " + str(np.max(diffimg)) + \
        "\nNumber of different pixels : " + str(np.size(diffimg)-np.sum(pixelimg)) + "\n> " + str(np.sum(pixelimg)) + "/" + str(np.size(diffimg)) + " = " + str(np.sum(pixelimg)/np.size(diffimg)*100)[:5] + "% unchanged"

    #print("\n- Mark :\nMean of difference :", str(np.mean(diffmark)), "\nMax of difference  :", str(np.max(diffmark)), 
    #      "\nNumber of different pixels :", str(np.size(diffmark)-np.sum(pixelmark))," --> ", str(np.sum(pixelmark)), "/", str(np.size(diffmark)), "=", str(np.sum(pixelmark)/np.size(diffmark)*100)[:5], "% unchanged")
    Mstats = "Mean of difference : " + str(np.mean(diffmark))[:6] + "\nMax of difference : " + str(np.max(diffmark)) + \
        "\nNumber of different pixels : " + str(np.size(diffmark)-np.sum(pixelmark)) + "\n> " + str(np.sum(pixelmark)) + "/" + str(np.size(diffmark)) + " = " + str(np.sum(pixelmark)/np.size(diffmark)*100)[:5] + "% unchanged"


    # In an new page plot diffimg and diffmark
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    # Adjust the height of the subplots
    fig.subplots_adjust(top=0.8, bottom=0.2)

    # Remove the ticks from the plots
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
        
    # Plot your images in each subplot
    axes[0].set_title("Difference between original and watermarked image (x15))")
    axes[0].imshow(diffimg*15)
    axes[0].set_xlabel(f'Size: {diffimg.shape}\n' + Istats)

    axes[1].set_title("Difference between original mark and recovered mark")
    axes[1].imshow(diffmark, cmap = 'gray')
    axes[1].set_xlabel(f'Size: {diffmark.shape}\n' + Mstats)

    plt.show()



######## Histogram ########

def plot_grayscale_histogram(image): 
    if type(image) == str:
        # read the image of a plant seedling as grayscale from the outset
        image = iio.imread(img, mode="L")
    # convert the image to float dtype with a value range from 0 to 1
    image = skimage.util.img_as_float(image)
    # create the histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    # configure and draw the histogram figure
    plt.figure()
    plt.figure(figsize=(13, 5))
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()

def plot_color_histogram(image):
    if type(image) == str:
        image = iio.imread(uri=image)
    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")
    
    # Histogram plot, with three lines, one for each color
    plt.figure()
    plt.figure(figsize=(13, 5))
    plt.xlim([0, 256])
    
    for channel_id, color in enumerate(colors):
        # create histogram for the current color
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)
    
    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    plt.show()
    
    
def plot_cie_ycbcr_y_histogram(image):
    if type(image) == str:
        image = iio.imread(uri=image)
    # Convert it to the CIE YCbCr color space
    image = skimage.color.rgb2ycbcr(image)
    # extract the Y channel
    image = image[:, :, 0]
    # create the histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    # configure and draw the histogram figure
    plt.figure()
    plt.figure(figsize=(13, 5))
    plt.title("CIE Y Histogram")
    plt.xlabel("CIE Y value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 256.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
    
def plot_cie_ycbcr_cb_histogram(image):
    if type(image) == str:
        image = iio.imread(uri=image)
    # Cconvert it to the CIE YCbCr color space
    image = skimage.color.rgb2ycbcr(image)
    # extract the Cb channel
    image = image[:, :, 1]
    # create the histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    # configure and draw the histogram figure
    plt.figure()
    plt.figure(figsize=(13, 5))
    plt.title("CIE Cb Histogram")
    plt.xlabel("CIE Cb value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 256.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
    
def plot_cie_ycbcr_cr_histogram(image):
    if type(image) == str:
        image = iio.imread(uri=image)
    # Convert it to the CIE YCbCr color space
    image = skimage.color.rgb2ycbcr(image)
    # extract the Cr channel
    image = image[:, :, 2]
    # create the histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    # configure and draw the histogram figure
    plt.figure()
    plt.figure(figsize=(13, 5))
    plt.title("CIE Cr Histogram")
    plt.xlabel("CIE Cr value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 256.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()

def plot_all_histograms(image):
    if type(image) == str:
        image = iio.imread(uri=image)
    plot_color_histogram(image)
    plot_grayscale_histogram(image)
    # luminessance
    plot_cie_ycbcr_y_histogram(image)
    # chrominance
    plot_cie_ycbcr_cb_histogram(image)
    plot_cie_ycbcr_cr_histogram(image)

    
    
    

################## Exemple ##################

### Parameters ###
""" 
x = 2 # Divise la taille de la marque

image = "original/leopard.jpg"
marque = "original/marque/dragon.jpg"
Iresult = "result/watermarkedImage.jpg"
Mresult = "result/recoveredWatermark.png"


### Main ###

plotResult(image, marque, Iresult, Mresult, x)

plotDiff(image, marque, Iresult, Mresult, x) 
"""