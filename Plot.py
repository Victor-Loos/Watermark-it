from matplotlib import pyplot as plt
import cv2
from Watermark import *


# Read the image and convert them in the same way as the watermarking process
def ImageProcessing(image, marque, Iresult, Mresult, x):
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
    marqueImg = Image.open(marque).resize((Wlength, Wlength), 1)
    marqueImg = marqueImg.convert('L')
    marqueImg = marqueImg.point(lambda x: 0 if x < 128 else 255, '1')

    # Result watermarked image :
    waterimg = cv2.imread(Iresult)
    waterimg = cv2.cvtColor(waterimg, cv2.COLOR_BGR2RGB)

    # Result recovered watermark :
    recovwater = cv2.imread(Mresult, cv2.IMREAD_GRAYSCALE)
    
    return readImage, usedImage, marqueImg, waterimg, recovwater


### Visualisation ###

#### Display results ####
def plotResult(readImage, marqueImg, waterimg, recovwater, x):
    readImage, usedImage, marqueImg, waterimg, recovwater  = ImageProcessing(readImage, marqueImg, waterimg, recovwater, x)
    
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
    readImage, usedImage, marqueImg, waterimg, recovwater = ImageProcessing(readImage, marqueImg, waterimg, recovwater, x)

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
