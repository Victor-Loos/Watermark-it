import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np

from watermark import *
from display import *


# Peak Signal to Noise Ratio
def compute_psnr(img1, img2):
    img2 = np.array(img2, dtype=np.uint8)
    
    img1 = img1.astype(float) / 255.
    img2 = img2.astype(float) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: # Means no noise in the signal .
        return "Same Image"
    return 10 * math.log10(1. / mse)


# Normalized cross correlation
def NCC(img1, img2):
    img2 = np.array(img2, dtype=np.uint8)
    
    img1 = img1.astype(float) / 255.
    img2 = img2.astype(float) / 255.
    return np.sum(img1 * img2) / (np.sqrt(np.sum(img1 ** 2)) * np.sqrt(np.sum(img2 ** 2)))
    



########## Attack ##########
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        #var = 0.001
        var = 0.08
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        #noisy = image + gauss
        noisy = image + image * gauss
        return noisy

    elif noise_typ == "s&p":
        row, col = image.shape[:2]  # get the row and column dimensions of the image
        s_vs_p = 0.5
        amount = 0.3
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                    for i in image.shape[:2]]  # generate random coordinates for each dimension separately
        out[coords[0], coords[1]] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                    for i in image.shape[:2]]  # generate random coordinates for each dimension separately
        out[coords[0], coords[1]] = 0

        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image)) 
        vals = 2 ** np.ceil(np.log2(vals))
        #noisy = np.random.poisson(image * vals) / float(vals) 
        noisy = np.random.poisson(image * vals) / float(vals) * 1.5
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        #noisy = image + image * gauss
        noisy = image + image * gauss*0.20
        return noisy
    elif noise_typ == "rotate90" :
        angle = 90
        scale = 1.0
        w = image.shape[1]
        h = image.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        noisy=cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        return noisy
    elif noise_typ=="chop30":
        img = image.copy()
        w, h = img.shape[:2]
        # chop off 30% of the top of the image
        #noisy=img[int(w * 0.3):, :] 
        # chop off 15% of the top of the image and 15% of the bottom
        noisy=img[int(w * 0.15):int(w * 0.85), :]
        return noisy
    
    elif noise_typ=="jpeg":
        # encode image as a jpeg with quality 40
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        # decode image
        decimg = cv2.imdecode(encimg, 1)
        decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
        decimg = cv2.cvtColor(decimg, cv2.COLOR_RGB2GRAY)
        return decimg
    
    elif noise_typ=="resize":
        # resize image to 50% of its original size
        noisy = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # inverse the resize operation
        noisy = cv2.resize(noisy, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return noisy
        """ # resize image to 25% of its original size
        noisy = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        # inverse the resize operation
        noisy = cv2.resize(noisy, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        return noisy """
        
        



###### Recover ######

def restoreCrop(img, originalImage):
    # Add null information to the borders of a cropped image to make it 2048x2048 in size:
    # Get the dimensions of the cropped image
    height, width = img.shape

    # Create a new black image of the same size as readImage
    new_img = np.zeros((originalImage.shape[0], originalImage.shape[1]), dtype=np.uint8)
    
    # Calculate the position to place the cropped image in the center of the new image
    x_offset = (new_img.shape[1] - width) // 2
    y_offset = (new_img.shape[0] - height) // 2
    
    # Add the cropped image to the center of the new image
    new_img[y_offset:y_offset+height, x_offset:x_offset+width] = img

    return new_img


###### Test attack ######

def attackAll(image, marque, Iresult, Mresult, x, password):
    ### Read images ###
    readImage, usedImage, marqueImg, waterimg, recovwater  = imageProcessing(image, marque, Iresult, Mresult, x)
    originalImage = usedImage
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    originalMarque = np.array(marqueImg, dtype=np.uint8)
    watermarked = waterimg
    watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)


    # Create a figure with 4 rows and 4 columns
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 9)) # (width, height)
    # Adjust the height of the subplots
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.4, wspace=0.2, left=0.01, right=0.99)
    # hspace and wspace are the space between subplots in height and width
    # top and bottom are the space between the figure and the subplots
    # left and right are the space between the figure and the subplots

    # Remove the ticks from the plots
    plt.setp(axes.flat, xticks=[], yticks=[])

    # Set the title of the matplotlib window that contains t

    # Set the title of the figure
    fig.suptitle("Image Watermarking Attacks", fontsize=16)

    # axes[row, col]

    ### Compression Attacks ###

    # Without attack
    extracted = recoverWatermark(Iresult, password)
    axes[0, 0].set_title("Default Jpeg compression (75%)")
    axes[0, 0].imshow(originalImage, cmap='gray')
    axes[0, 1].set_title("Recovered Watermark")
    axes[0, 1].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, watermarked)
    NCC2 = NCC(originalMarque, extracted)
    axes[0, 0].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[0, 1].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)

    # jpeg
    img = noisy("jpeg", watermarked)
    cv2.imwrite("attack/jpeg.jpg", img) 
    extracted = recoverWatermark("attack/jpeg.jpg", password)
    axes[1, 0].set_title("Jpeg 40% Image")
    axes[1, 0].imshow(img, cmap='gray')
    axes[1, 1].set_title("Extracted Watermark")
    axes[1, 1].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[1, 0].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[1, 1].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)


    ### Geometric Attacks ###
    # Rotate 90°
    img = noisy("rotate90", watermarked)
    cv2.imwrite("attack/rotate90.jpg", img) 
    extracted = recoverWatermark("attack/rotate90.jpg", password)
    axes[2, 0].set_title("Rotate 90° Image")
    axes[2, 0].imshow(img, cmap='gray')
    axes[2, 1].set_title("Extracted Watermark")
    axes[2, 1].imshow(extracted, cmap='gray')
    NCC2 = NCC(originalMarque, extracted)
    axes[2, 1].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)

    # Chop 30%
    img = noisy("chop30", watermarked)
    img = restoreCrop(img, originalImage)
    cv2.imwrite("attack/chop30.jpg", img)
    extracted = recoverWatermark("attack/chop30.jpg", password)
    axes[3, 0].set_title("Chop 30% Image")
    axes[3, 0].imshow(img, cmap='gray')
    axes[3, 1].set_title("Extracted Watermark")
    axes[3, 1].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[3, 0].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[3, 1].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)



    ### Image Filtering Attacks ###
    
    # Gaussian Noise
    img = noisy("gauss", watermarked)
    cv2.imwrite("attack/gaussian.jpg", img)
    extracted = recoverWatermark("attack/gaussian.jpg", password)
    axes[0, 2].set_title("Gaussian Noise Image")
    axes[0, 2].imshow(img, cmap='gray')
    axes[0, 3].set_title("Extracted Watermark")
    axes[0, 3].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[0, 2].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[0, 3].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)

    # Salt and Pepper Noise
    img = noisy("s&p", watermarked)
    cv2.imwrite("attack/salt_pepper.jpg", img)
    extracted = recoverWatermark("attack/salt_pepper.jpg", password)
    axes[1, 2].set_title("Salt and Pepper Image")
    axes[1, 2].imshow(img, cmap='gray')
    axes[1, 3].set_title("Extracted Watermark")
    axes[1, 3].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[1, 2].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[1, 3].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)

    # Poisson Noise
    img = noisy("poisson", watermarked)
    cv2.imwrite("attack/poisson.jpg", img)
    extracted = recoverWatermark("attack/poisson.jpg", password)
    axes[2, 2].set_title("Poisson Noise Image")
    axes[2, 2].imshow(img, cmap='gray')
    axes[2, 3].set_title("Extracted Watermark")
    axes[2, 3].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[2, 2].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[2, 3].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)

    # Speckle Noise
    img = noisy("speckle", watermarked)
    cv2.imwrite("attack/speckle.jpg", img)
    extracted = recoverWatermark("attack/speckle.jpg", password)
    axes[3, 2].set_title("Speckle Noise Image")
    axes[3, 2].imshow(img, cmap='gray')
    axes[3, 3].set_title("Extracted Watermark")
    axes[3, 3].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[3, 2].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[3, 3].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)


    plt.show()
    
    
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 9)) # (width, height)
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.4, wspace=0.2, left=0.01, right=0.99)
    # Remove the ticks from the plots
    plt.setp(axes.flat, xticks=[], yticks=[])
    # Set the title of the figure
    fig.suptitle("Image Watermarking Attacks P2", fontsize=16)
    
    # Wrong Password
    extracted = recoverWatermark(Iresult, "wrongPassword")
    axes[0, 0].set_title("Wrong Password")
    axes[0, 0].imshow(originalImage, cmap='gray')
    axes[0, 1].set_title("Recovered Watermark")
    axes[0, 1].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, watermarked)
    NCC2 = NCC(originalMarque, extracted)
    axes[0, 0].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[0, 1].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)

    # Resize Image
    img = noisy("resize", watermarked)
    cv2.imwrite("attack/resize.jpg", img) 
    extracted = recoverWatermark("attack/resize.jpg", password)
    axes[1, 0].set_title("Resize 50% Image")
    axes[1, 0].imshow(img, cmap='gray')
    axes[1, 1].set_title("Extracted Watermark")
    axes[1, 1].imshow(extracted, cmap='gray')
    psnr1 = compute_psnr(originalImage, img)
    NCC2 = NCC(originalMarque, extracted)
    axes[1, 0].set_xlabel(f"PSNR : {psnr1:.2f}", fontsize=8)
    axes[1, 1].set_xlabel(f"NCC : {NCC2:.2f}", fontsize=8)
    
    plt.show()
    


#### Plot individual images ####
def attackOne(image, marque, Iresult, Mresult, x, password):
    ### Read images ###
    readImage, usedImage, marqueImg, waterimg, recovwater  = imageProcessing(image, marque, Iresult, Mresult, x)
    originalImage = usedImage
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    originalMarque = np.array(marqueImg, dtype=np.uint8)
    watermarked = waterimg
    watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    
    print("Without attack")
    extracted = recoverWatermark(Iresult, password)
    fig = plt.figure("Without attack")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(originalImage, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Original Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Recovered Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,watermarked))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))

    print("\n\n")
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\t\t\t\t Compression Attacks')
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
    img = noisy( "jpeg",watermarked)
    cv2.imwrite("attack/jpeg.jpg", img) 
    extracted = recoverWatermark("attack/jpeg.jpg", password)
    #cv2.imwrite("attack/jpeg.png", extracted)
    fig = plt.figure("Jpeg 45%")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Jpeg 45% Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))

    print("\n\n")
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\t\t\t\t Geometric Attacks')
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

    print ("\t\t\t\t Rotate 90°")
    print ("-----------------------------------------------------------------------------------------")
    img = noisy( "rotate90",watermarked)
    cv2.imwrite("attack/rotate90.jpg", img) 
    extracted = recoverWatermark("attack/rotate90.jpg", password)
    fig = plt.figure("Rotate 90°")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Rotate 90° Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    #print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))

    print ("******************************************************************************************")
    print ("\t\t\t\t Chop 30")
    print ("-----------------------------------------------------------------------------------------")
    img = noisy( "chop30",watermarked)
    img = restoreCrop(img, originalImage)
    cv2.imwrite("attack/chop30.jpg", img)
    extracted = (recoverWatermark("attack/chop30.jpg", password)) # Image.fromarray(recoverWatermark(img))
    fig = plt.figure("chop 30")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("chop 30 Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))
    print ("******************************************************************************************")

    print("\n\n")
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\t\t\t\t Image Filtering Attacks')
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

    print ("\t\t\t\t Gaussian Noise ")
    print ("-----------------------------------------------------------------------------------------")
    img = noisy( "gauss",watermarked)
    cv2.imwrite("attack/gauss.jpg", img)
    extracted = (recoverWatermark("attack/gauss.jpg", password))
    fig = plt.figure("Gaussian Noise")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Gaussian Noise Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))
    print ("******************************************************************************************")

    print ("\t\t\t\t Salt and Pepper Noise")
    print ("-----------------------------------------------------------------------------------------")
    img = noisy( "s&p",watermarked)
    cv2.imwrite("attack/s&p.jpg", img)
    extracted = (recoverWatermark("attack/s&p.jpg", password))
    fig = plt.figure("Salt and Pepper Noise")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Salt and Pepper Noise Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))
    print ("******************************************************************************************")

    print ("\t\t\t\t Poisson Noise")
    print ("-----------------------------------------------------------------------------------------")
    img = noisy( "poisson",watermarked)
    cv2.imwrite("attack/poisson.jpg", img)
    extracted = (recoverWatermark("attack/poisson.jpg", password))
    fig = plt.figure("Poisson Noise")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Poisson Noise Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))
    #print ("\t\t\t NCC ", NCC(originalImage, img))
    #print ("\t\t\t NCC ", NCC(originalMarque, extracted))
    print ("******************************************************************************************")

    print ("\t\t\t\t Speckle Noise")
    print ("-----------------------------------------------------------------------------------------")
    img = noisy( "speckle",watermarked)
    cv2.imwrite("attack/speckle.jpg", img)
    extracted = (recoverWatermark("attack/speckle.jpg", password))
    fig = plt.figure("Speckle Noise")
    fig.set_size_inches(10, 5)
    plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Speckle Noise Image")
    plt.subplot(1,2,2), plt.imshow(extracted, cmap = 'gray', vmin = 0, vmax = 255), plt.title("Extracted Watermark")
    print("\t\t\t PSNR de l'image :", compute_psnr(originalImage,img))
    print("\t\t\t PSNR de la marque :", compute_psnr(originalMarque,extracted))
    print ("******************************************************************************************")

    plt.show()
    
    
    
########## Main ##########

### Parameters ###
""" 
x = 2 # Divise la taille de la marque
password = "my_password"

image = "original/leopard.jpg"
marque = "original/marque/dragon.jpg"
Iresult = "result/watermarkedImage.jpg"
Mresult = "result/recoveredWatermark.png"


attackAll(image, marque, Iresult, Mresult, x, password)
#attackOne(image, marque, Iresult, Mresult, x, password)
"""
