import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy.fftpack import dct
from scipy.fftpack import idct

from typing import List
import hashlib
import struct



##### Function Convert and resize image #####

def convertImage(imageName):
    # Read the image and convert it to RGB
    img = cv2.imread(imageName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get the image dimensions and resize it if necessary
    height, width = img.shape[:2]
    # Resize to have minimum 2048 in width or height with the ratio of the original image
    if height < 2048 and width < 2048:
        if height > width:
            width = math.ceil(width*2048/height)
            height = 2048
        else:
            height = math.ceil(height*2048/width)
            width = 2048   
        img = cv2.resize(img, (width, height))         
        
    if height % 8 != 0 or width % 8 != 0:
        height = math.ceil(height/8)*8
        width = math.ceil(width/8)*8
        img = cv2.resize(img, (width, height))
    
    # Convert the RGB image to YCbCr and split it into its three channels
    imgC = Image.fromarray(img)
    imgYCbCr = imgC.convert('YCbCr')
    Y, Cb, Cr = imgYCbCr.split()
    
    # Convert the Y channel to a numpy array
    imageArray = np.array(Y.getdata(), dtype=float).reshape((imgYCbCr.size[1], imgYCbCr.size[0]))
    
    # Return the numpy array, the color channels, and the image dimensions
    return imageArray, (Y, Cb, Cr), (width, height)


# Convert the image to binary image of 1:1 less or equal to Msize
def convertMark(imageName, Msize):
    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    size = img.shape
    # If it not already Look for the nearest multiple of 8 for height and width
    if size[0] % 8 != 0 or size[1] % 8 != 0:
        size = (math.ceil(size[1]/8)*8, math.ceil(size[0]/8)*8)
    # Check if the image is not too big for the matrix M
    if size[0] > Msize[0] or size[1] > Msize[1]:
        # Search the size for the image to be equal or less than the matrix M size with a ratio of 1
        if size[0] > Msize[0]:
            size = (Msize[0], Msize[0])
        if size[1] > Msize[1]:
            size = (Msize[1], Msize[1])
            
    # Resize the image
    img = cv2.resize(img, size)
    
    # Converting the gray scale image to binary image
    img = cv2.threshold(img, 128, maxVal, cv2.THRESH_BINARY)[1]
    markArray = np.array(img, dtype=float).reshape((size[0], size[1]))
    return markArray


## Convert text 

def utf8_to_binary(text: str):
    # Convert the text to a bytearray in UTF-8 encoding
    byte_array = bytearray(text.encode("utf-8"))

    # Convert each byte to binary and concatenate them into a string
    binary_string = "".join(f"{byte:08b}" for byte in byte_array)
    return binary_string
                          

def binary_list_to_utf8(binary_list: List[int]) -> str:
    # Inverse the conversion of 1 to 255 and 0 to 0
    binary_list = [0 if x < 1 else 1 for x in binary_list]
    # Convert the binary list to a binary string
    binary_string = "".join(map(str, binary_list))
    # Convert the binary string to a bytearray and then to a string in UTF-8 encoding
    byte_array = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
    # Convert the binary string to a bytearray and then to a string in UTF-8 encoding,
    # replacing unknown characters with a dash
    try:
        byte_array = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
        text = byte_array.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        byte_array = bytearray(int(binary_string[i:i+8], 2) if binary_string[i:i+8] != "00000000" else 45 for i in range(0, len(binary_string), 8))
        text = byte_array.decode("utf-8", errors="replace")
    return text


### DCT and IDCT

# Function that apply Discrete Cosine Transform on an 8x8 block
def applyDCT(imageArray):
    rows, cols = imageArray.shape
    allSubdct = np.empty((rows, cols))
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            subpixels = imageArray[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            allSubdct[i:i+8, j:j+8] = subdct
    return allSubdct
    
    
# Function that apply Inverse Discrete Cosine Transform on an 8x8 block
def inverseDCT(allSubdct):
    rows, cols = allSubdct.shape
    allSubidct = np.empty((rows, cols))
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            subidct = idct(idct(allSubdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            allSubidct[i:i+8, j:j+8] = subidct
    return allSubidct


### Coordinate generation

def password_to_position(password, Msize, nbCoords):
    width, height = Msize[0], Msize[1]
    #nbCoords = width * height
    max_coord = width * height
    # required bits to code x,y coordinates of image
    len_x = (width - 1).bit_length()
    len_y = (height - 1).bit_length()
    # Mask for bitwise AND operations instead of modulo operations
    mask_x = 2 ** len_x - 1
    mask_y = 2 ** len_y - 1

    password_bytes = password.encode('utf-8')
    pwHash = hashlib.sha256(password_bytes).digest()
    j = 0

    coords = []
    while len(coords) < nbCoords:
        if j + 8 > len(pwHash):
            pwHash = hashlib.sha256(pwHash).digest()
            j = 0
        # <2I for 2 unsigned 32-bit integers in little-endian format.
        x, y = struct.unpack_from('<2I', pwHash, j) 
        j += 8
        x, y = x & mask_x, y & mask_y
        index = (x + y * width) % max_coord
        if index not in coords:
            coords.append(index)

    return [(i % width, i // width) for i in coords]

def generate_coordinates(Msize, nbCoords):
    width, height = Msize
    coords = []
    x, y = 0, 0
    for i in range(int(nbCoords)):
        coords.append((x, y))
        x += 1
        if x == width:
            x = 0
            y += 1
        if y == height:
            break
    return coords


################## Function that embedded watermark image into cover image ##################

# Embed watermark into the cover image with frequency-flipping method
""" 
C1 >  C2 --> 0
C1 <= C2 --> 1
"""
def embedWatermark(watermarkArray, pixel_positions, originalImage):
    watermarkFlat = watermarkArray.ravel() # ravel() is used to convert 2D array to 1D array
    ind = 0
    for x, y in pixel_positions:
        if ind < len(watermarkFlat):
            subdct = originalImage[x*8:x*8+8, y*8:y*8+8]
            # Check if the pixel is 0 or 255
            if watermarkFlat[ind] == 0:
                # If the pixel is 0 we want C1 > C2
                if subdct[C1x][C1y] < subdct[C2x][C2y]+maxVal:
                    # Flip the frequency and add the maxVal
                    subdct[C1x][C1y], subdct[C2x][C2y] = subdct[C2x][C2y], subdct[C1x][C1y]
                    if subdct[C1x][C1y] < subdct[C2x][C2y]+maxVal*10:
                        subdct[C1x][C1y] += maxVal
                        subdct[C2x][C2y] -= maxVal
            else:
                if subdct[C1x][C1y] >= subdct[C2x][C2y]-maxVal:
                    # Flip the frequency
                    subdct[C1x][C1y], subdct[C2x][C2y] = subdct[C2x][C2y], subdct[C1x][C1y]
                    if subdct[C1x][C1y] >= subdct[C2x][C2y]-maxVal*10:
                        subdct[C1x][C1y] -= maxVal
                        subdct[C2x][C2y] += maxVal
            originalImage[x*8:x*8+8, y*8:y*8+8] = subdct
            ind += 1
    return originalImage  


## Function that links the different functions for watermark insertion

def embeddedImage(coverImage, watermarkImage, password=None): #Todo
    imageArray, colors, size = convertImage(coverImage)
    Isize = (int(size[1]/2/8), int(size[0]/2/8)) # Available space for watermark
    Wsize = (int(size[1]/2/8/x), int(size[0]/2/8/x)) # Watermark size
    watermarkArray = convertMark(watermarkImage, Wsize)
    nbCoords = watermarkArray.shape[0] * watermarkArray.shape[1]
    
    if password is not None:
        pixel_positions = password_to_position(password, Isize, nbCoords)
    else :
        pixel_positions = generate_coordinates(Isize, nbCoords)

    coeffsImage = list(pywt.wavedec2(data=imageArray, wavelet = 'haar', level = 1))
    dctArray = applyDCT(coeffsImage[0]) # [0] corresponds to cH subband (cH = LH)    

    dctArray = embedWatermark(watermarkArray, pixel_positions, dctArray)

    coeffsImage[0] = inverseDCT(dctArray)
    imageArrayH=pywt.waverec2(coeffsImage, 'haar')
    
    # Get Y Cb Cr channels
    yOriginal, Cb, Cr = colors
    y = imageArrayH
    
    # Convert Cb and Cr images to numpy arrays
    Cb_arr = np.array(Cb)
    Cr_arr = np.array(Cr)

    # Cap the values of the Y array between 0 and 255
    y = np.clip(y, 0, 255) # white and black pixels error

    # Convert y array to uint8 rounded down
    y = np.uint8(y)

    # Stack Y, Cb, and Cr arrays together
    YCbCr_arr = np.dstack((y, Cb_arr, Cr_arr))

    # Convert YCbCr array to PIL Image object
    YCbCr_img = Image.fromarray(YCbCr_arr.astype(np.uint8), mode='YCbCr')

    return YCbCr_img


def embeddedTexte(coverImage, texte, password=None):
    texte = texte + '---'
    Tsize = int((len(texte)+7)*8)
    imageArray, colors, size = convertImage(coverImage)
    Isize = (int(size[1]/2/8), int(size[0]/2/8)) # Available space for watermark
    
    # Maximal size of the text
    nbCoords = (Isize[0]/x)*(Isize[1]/x)
    if Tsize > nbCoords:
        Tsize = nbCoords
    
    texteArray = utf8_to_binary(texte)
    # Convert string to list of characters
    charList = list(texteArray)
    # Convert list of characters to numpy array of int64
    intArray = np.array(charList, dtype=np.int64)
    
    if password is not None:
        pixel_positions = password_to_position(password, Isize, Tsize)
    else:
        pixel_positions = generate_coordinates(Isize, Tsize)
    
    coeffsImage = list(pywt.wavedec2(data=imageArray, wavelet = 'haar', level = 1))
    dctArray = applyDCT(coeffsImage[0])
    
    dctArray = embedWatermark(intArray, pixel_positions, dctArray)

    coeffsImage[0] = inverseDCT(dctArray)
    imageArrayH=pywt.waverec2(coeffsImage, 'haar')
    
    # Get Y Cb Cr channels
    yOriginal, Cb, Cr = colors
    y = imageArrayH
    
    # Convert Cb and Cr images to numpy arrays
    Cb_arr = np.array(Cb)
    Cr_arr = np.array(Cr)

    # Cap the values of the Y array between 0 and 255
    y = np.clip(y, 0, 255) # white and black pixels error

    # Convert y array to uint8 rounded down
    y = np.uint8(y)

    # Stack Y, Cb, and Cr arrays together
    YCbCr_arr = np.dstack((y, Cb_arr, Cr_arr))

    # Convert YCbCr array to PIL Image object
    YCbCr_img = Image.fromarray(YCbCr_arr.astype(np.uint8), mode='YCbCr')
    return YCbCr_img



################## Function that extract watermark image from embedded image ##################

# Get watermark from the cover image with frequency-flipping method
def getWatermark(dctWatermarkedCoeff, pixel_positions, watermarkSize=None):
    subwatermarks = []
    for x, y in pixel_positions:
        coeffSlice = dctWatermarkedCoeff[x*8:x*8+8, y*8:y*8+8]
        if coeffSlice[C1x][C1y] > coeffSlice[C2x][C2y]:
            subwatermarks.append(0)
        else:
            subwatermarks.append(1)
    if watermarkSize is None: # For the text watermark
        return subwatermarks
    watermark = np.array(subwatermarks).reshape(watermarkSize, watermarkSize) 
    watermark = watermark * 255   
    return watermark


## Function that links the different functions for watermark extraction

def recoverWatermark(image, password=None, Wsize=None):
    # Get the necessary parameters :
    imageArray, colors, size = convertImage(image)
    Isize = (int(size[1]/2/8), int(size[0]/2/8)) # Available space for watermark
    if Wsize != None:
        nbCoords = Wsize[0] * Wsize[1]
    else:
        # Size of the watermark square
        Wsize = (int(Isize[0]/x), int(Isize[1]/x)) 
        # Compare width and height of Msize and multiply the smallest
        if Wsize[0] < Wsize[1]:
            nbCoords = Wsize[0] * Wsize[0]
        else:
            nbCoords = Wsize[1] * Wsize[1]
    Wlength = int(np.sqrt(nbCoords))
    
    # Start the recovery process :
    coeffsWatermarkedImage=list(pywt.wavedec2(data = imageArray, wavelet = 'haar', level = 1))
    dctWatermarkedCoeff = applyDCT(coeffsWatermarkedImage[0])
    
    if password is not None:
        pixel_positions = password_to_position(password, Isize, nbCoords)
    else:
        pixel_positions = generate_coordinates(Isize, nbCoords)
    
    watermarkArray = getWatermark(dctWatermarkedCoeff, pixel_positions, Wlength)
    watermarkArray =  np.uint8(watermarkArray)
    watermarkArray = Image.fromarray(watermarkArray)
    return watermarkArray


def recoverText(image, password=None, nbChr=None):
    # Get the necessary parameters :
    imageArray, colors, size = convertImage(image)
    Isize = (size[1]//16, size[0]//16) # Available space for watermark
    Wsize = [Isize[0]//x, Isize[1]//x] # Watermark size

    if nbChr is not None:
        nbCoords = nbChr * 8
    else:
        """ # Size of the watermark square multiple of 8 
        while int(Wsize[0]) % 8 != 0:
            Wsize[0] -= 1
        while int(Wsize[1]) % 8 != 0:
            Wsize[1] -= 1 """
        nbCoords = Wsize[0] * Wsize[1]
        
    
    # Start the recovery process :
    coeffsWatermarkedImage=list(pywt.wavedec2(data = imageArray, wavelet = 'haar', level = 1))
    dctWatermarkedCoeff = applyDCT(coeffsWatermarkedImage[0])
    
    if password is not None:
        pixel_positions = password_to_position(password, Isize, nbCoords)
    else: 
        pixel_positions = generate_coordinates(Isize, nbCoords)
    
    watermarkArray = getWatermark(dctWatermarkedCoeff, pixel_positions, None)
    # If the watermark is not in list format convert it
    if type(watermarkArray) != list:
        watermarkArray = [int(j) for j in watermarkArray.flatten()] 
    watermarkArray = binary_list_to_utf8(watermarkArray)
    
    # Look for dash in the text to know where the watermark ends if 3 dashes are found next to each other cut the text
    dashCount = 0
    for i in range (0, len(watermarkArray)-1, 1):
        if watermarkArray[i] == '-':
            dashCount += 1
        else:
            dashCount = 0
        if dashCount == 3:
            watermarkArray = watermarkArray[:i-2]
            break
    return watermarkArray



############################################ Main ############################################

############## Parameters ##############
# Différence de valeur des coeff de la DCT de l'image 
maxVal = 20 # recomendation : 1 à 50 
""" 0-8 -> coefficient des bloc 8x8 de la DCT à utiliser
Par défaut :
C(4,1) >  C(2,3) --> 0
C(4,1) <= C(2,3) --> 1
"""
C1x, C1y = 4, 1 
C2x, C2y = 2, 3
password = "my_password"

# Taille de la matrice utilisée pour la marque 
x = 2 # Divise la taille de la marque
# Taille de l'image/2/8/x -> Dct2D/Bloc8x8/x

# Chemin des images
image = "original/cascade.jpg"
marque = "original/marque/dragon.jpg"
Iresult = "result/watermarkedImage.jpg"
Mresult = "result/recoveredWatermark.png"

# Texte à insérer
texte = "Le tatouage numérique (digital watermark, « filigrane numérique ») est une technique permettant d'ajouter des informations +++ les textes trop longs sont automatiquement coupés"
#texte = "Test d'insertion de text  !"


############## Watermarking ##############

###### Example of use ######

## Watermarking with image

watermarkeImage = embeddedImage(image, marque, password) 
watermarkeImage.save(Iresult)

watermarkArray = recoverWatermark(Iresult, password) # Wsize=(x, y) for specific mark
watermarkArray.save(Mresult)


## Watermarking with text
""" 
watermarkeImage = embeddedTexte(image,texte, password)
watermarkeImage.save(Iresult)

watermarkArray = recoverText(Iresult, password)
print(watermarkArray) 
"""