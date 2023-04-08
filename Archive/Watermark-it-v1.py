import numpy as np
import pywt
import cv2
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct

from typing import List
import hashlib

###
# pip install PyWavelets
# pip install opencv-python
# pip install Pillow
# pip install scipy
###
# Tested on Python 3.8.16 64-bit, Python 3.9.16 64-bit, Python 3.10.9 64-bit
###

############### Function Convert and resize image ###############
def convertImage(imageName, size):
    img = Image.open(imageName).resize((size, size), 1) 
    # Convert image to jpeg if it is not
    """ if (img.format != 'JPEG'):
        img = img.convert('RGB')
        img.save('temp.jpg')
        img = Image.open('temp.jpg') """

    # Convert RGB image to YCbCr
    img = img.convert('YCbCr')
    # Split the YCbCr image into Y, Cb and Cr channels
    colors = img.split()
    # Convert the channel to a numpy array
    y = np.array(colors[0])
    Cb = np.array(colors[1])
    Cr = np.array(colors[2])
    img = y
    
    # Convert the numpy array to a PIL image
    img = Image.fromarray(img)
    imageArray = np.array(img.getdata(), dtype=float).reshape((size, size))
    return imageArray, colors


def convertMark(imageName, size):
    mark = Image.open(imageName).resize((size, size), 1)
    # Convert RGB image to gray scale
    mark = mark.convert('L')
    # Converting the gray scale image to binary image
    mark = mark.point(lambda x: 0 if x < 128 else maxVal, '1')
    # Inverting the binary image if there are more 255s than 0s
    if (np.sum(mark) > (size * size / 2)):
        mark = mark.point(lambda x: 0 if x == maxVal else maxVal, '1')

    markArray = np.array(mark.getdata(), dtype=float).reshape((size, size))
    return markArray

### Convert text

def utf8_to_binary_matrix(text: str) -> np.ndarray: #todo! renvoie que des 0 ?? 
    # Convert the text to a bytearray in UTF-8 encoding
    byte_array = bytearray(text.encode("utf-8"))

    # Convert each byte to binary and concatenate them into a string
    binary_string = "".join(f"{byte:08b}" for byte in byte_array)
        
    # Create a numpy matrix array of 32x32 padded with 0
    binary_matrix = np.full((32, 32), 0, dtype=int)

    # Put the binary string into the matrix
    for i in range(len(binary_string)):
        binary_matrix[i // 32][i % 32] = int(binary_string[i])
    
    # Convert 1 to 255 and 0 to 0    
    binary_matrix = binary_matrix * (maxVal*3.6)
    return binary_matrix


def binary_list_to_utf8(binary_list: List[int]) -> str:
    # Inverse the conversion of 1 to 255 and 0 to 0
    binary_list = [0 if x < (maxVal*2.3) else 1 for x in binary_list]
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


### Embed watermark into the cover image
def embedWatermark(watermarkArray, originalImage):
    watermarkArraySize = len(watermarkArray[0])
    watermarkFlat = watermarkArray.ravel() # ravel() is used to convert 2D array to 1D array
    ind = 0

    for x in range (0, len(originalImage), 8):
        for y in range (0, len(originalImage), 8):
            if ind < len(watermarkFlat): 
                subdct = originalImage[x:x+8, y:y+8]
                # Embed the fingerprint pixels into mid-Highfrequency cell of the 8x8 block.
                subdct[DctCoef[0]][DctCoef[1]] = watermarkFlat[ind]
                originalImage[x:x+8, y:y+8] = subdct
                ind+= 1
    return originalImage

### DCT transform on image, i.e. image array
def applyDCT(imageArray):
    size = len(imageArray[0])
    allSubdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = imageArray[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            allSubdct[i:i+8, j:j+8] = subdct
    return allSubdct

def inverseDCT(allSubdct):
    size = len(allSubdct[0])
    allSubidct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(allSubdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            allSubidct[i:i+8, j:j+8] = subidct

    return allSubidct

def getWatermark(dctWatermarkedCoeff, watermarkSize):
    subwatermarks = []
    for x in range (0, len(dctWatermarkedCoeff), 8):
        for y in range (0, len(dctWatermarkedCoeff), 8):
            coeffSlice = dctWatermarkedCoeff[x:x+8, y:y+8]
            subwatermarks.append(coeffSlice[DctCoef[0]][DctCoef[1]])
    watermark = np.array(subwatermarks).reshape(watermarkSize, watermarkSize)
    return watermark


def password_to_position(password):
    """ Here is an implementation of a Python function that takes a password 
    as a parameter and returns a 1024-pixel position in a 128x128 grid: """
    # Convert password to a string of bytes
    password_bytes = password.encode('utf-8')
    # Initialize the hash with the password bytes
    hash_bytes = password_bytes
    hash_bytes = hashlib.sha256(hash_bytes).digest()
    # Repeat the hashing process 32 times to generate 1024 integers
    # 256/8 * 32 = 32 * 32 = 1024
    x_hash_bytes = b''
    y_hash_bytes = b''
    for i in range(32+6): # +6 to be sure to have 1024 different pixels
        # Compute the SHA-256 hash of the previous hash bytes
        x_hash_bytes += hashlib.sha256(hash_bytes+x_hash_bytes).digest()
        y_hash_bytes += hashlib.sha256(x_hash_bytes).digest()
        
    # Convert the hash bytes to a list of integers
    x_hash_ints = [int(byte) for byte in x_hash_bytes]
    y_hash_ints = [int(byte) for byte in y_hash_bytes]
    # Map each integer to a position in a 128x128 grid
    x_positions = [position % 128 for position in x_hash_ints]
    y_positions = [position % 128 for position in y_hash_ints]
    pixel_positions = list(zip(x_positions, y_positions))

    # Remove duplicates
    pixel_positions = list(set(pixel_positions))
    return pixel_positions

def marqueToPosition(watermarkArray, pixel_positions):
    # watermarkArray : 32x32 array of the watermark
    # pixel_positions : list of the pixel positions in tuple (x, y)
    iw, jw = 0, 0
    inserted = np.zeros((128, 128))
    # Put watermarkArray in the matrix for the pixel_positions
    for x, y in pixel_positions:
        inserted[x][y] = watermarkArray[iw][jw]
        iw += 1
        if iw == len(watermarkArray[jw]):
            iw = 0
            jw += 1
            if jw == len(watermarkArray):
                break
    return inserted

def marqueFromPosition(watermarkArray, pixel_positions):
    # watermarkArray : 128x128 array with the watermark
    # pixel_positions : list of the pixel positions in tuple (x, y)
    iw, jw = 0, 0
    extracted = np.zeros((32, 32))
    # Get the pixel in watermarkArray form pixel_positions
    for x, y in pixel_positions:
        extracted[iw][jw] = watermarkArray[x][y]
        iw += 1
        if iw == len(extracted[jw]):
            iw = 0
            jw += 1
            if jw == len(extracted[iw]):
                break
    return extracted

def jpegCorrection(watermarkArray):
    # Convertion de toutes les valeur entre 40 et 60 en 255 et le reste en 0
    watermarkArray[watermarkArray > maxVal+13] = 255 # Blanc
    watermarkArray[watermarkArray < maxVal-13] = 255
    watermarkArray[watermarkArray != 255] = 0 # Noir -> Marque
    return watermarkArray

def printImage(imageArray):
    imageArrayCopy = imageArray.clip(0, 255)
    imageArrayCopy = imageArrayCopy.astype("uint8")
    img = Image.fromarray(imageArrayCopy)
    #img.save(name)
    return img


def recoverWatermark(img, password, outputImageName=None):
    watermarkeImage = cv2.imread(img)
    watermarkeImage = cv2.cvtColor(watermarkeImage, cv2.COLOR_BGR2YCrCb)
    watermarkeImage = watermarkeImage.astype(np.float32)
    y = watermarkeImage[:,:,0]
    Cb = watermarkeImage[:,:,1]
    Cr = watermarkeImage[:,:,2]
    # Select the channel
    image = y

    coeffsWatermarkedImage=list(pywt.wavedec2(data = image, wavelet = 'haar', level = 1))
    dctWatermarkedCoeff = applyDCT(coeffsWatermarkedImage[0])
    watermarkArray = getWatermark(dctWatermarkedCoeff, 128)

    watermarkArray =  np.uint8(watermarkArray)
    #watermarkArray = threshold(watermarkArray)    

    pixel_positions = password_to_position(password)
    watermarkArray = marqueFromPosition(watermarkArray, pixel_positions)

    watermarkArray = jpegCorrection(watermarkArray)

    #Save result
    """ if outputImageName != None:
        printImage(watermarkArray, outputImageName)
    else:
        printImage(watermarkArray, 'recoveredWatermark.png') """
    watermarkArray = printImage(watermarkArray)
    return watermarkArray

def recoverText(img, password):
    watermarkeImage = cv2.imread(img)
    watermarkeImage = cv2.cvtColor(watermarkeImage, cv2.COLOR_BGR2YCrCb)
    watermarkeImage = watermarkeImage.astype(np.float32)
    y = watermarkeImage[:,:,0]
    Cb = watermarkeImage[:,:,1]
    Cr = watermarkeImage[:,:,2]
    # Select the channel
    image = y

    coeffsWatermarkedImage=list(pywt.wavedec2(data = image, wavelet = 'haar', level = 1))
    dctWatermarkedCoeff = applyDCT(coeffsWatermarkedImage[0])
    
    # Fonction getText à ajouter ! pour retourner une liste et la bonne taille 
    watermarkArray = getWatermark(dctWatermarkedCoeff, 128)
    
    pixel_positions = password_to_position(password) 
    watermarkArray = marqueFromPosition(watermarkArray, pixel_positions) # todo : voir la coresponance entre les deux
    
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
        
    #Save result in txt file
    """ else :
        file = open('result/recoveredText.txt', 'w')
        file.write(watermarkArray)
        file.close() """
    return watermarkArray


def embeddedImage(coverImage, watermarkImage, password):
    imageArray, colors = convertImage(coverImage, 2048)
    watermarkArray = convertMark(watermarkImage, 32) # 128/4
    
    pixel_positions = password_to_position(password)
    watermarkArray = marqueToPosition(watermarkArray, pixel_positions)

    coeffsImage = list(pywt.wavedec2(data=imageArray, wavelet = 'haar', level = 1))
    dctArray = applyDCT(coeffsImage[0]) # [0] corresponds to cH subband (cH = LH)
    dctArray = embedWatermark(watermarkArray, dctArray)

    coeffsImage[0] = inverseDCT(dctArray)
    imageArrayH=pywt.waverec2(coeffsImage, 'haar')
    
    # Get Y Cb Cr channels
    y, Cb, Cr = colors
    # Convert the numpy array back to an image
    y = Image.fromarray(imageArrayH.astype('uint8'))

    # Put the red channel back in the RGB image
    colors = (y, Cb, Cr)

    # Save result
    img = Image.merge('YCbCr', colors).convert('RGB')
    """ img.save('result/watermarkedImage.jpg')
    img.save('result/watermarkedImage.png') """    
    return img


def embeddedTexte(coverImage, texte, password):
    imageArray, colors = convertImage(coverImage, 2048)
    texteArray = utf8_to_binary_matrix(texte)
    
    pixel_positions = password_to_position(password)
    texteArray = marqueToPosition(texteArray, pixel_positions)

    coeffsImage = list(pywt.wavedec2(data=imageArray, wavelet = 'haar', level = 1))
    dctArray = applyDCT(coeffsImage[0])
    dctArray = embedWatermark(texteArray, dctArray)

    coeffsImage[0] = inverseDCT(dctArray)
    imageArrayH=pywt.waverec2(coeffsImage, 'haar')
    
    # Get Y Cb Cr channels
    y, Cb, Cr = colors
    # Convert the numpy array back to an image
    y = Image.fromarray(imageArrayH.astype('uint8'))

    # Put the red channel back in the RGB image
    colors = (y, Cb, Cr)

    # Save result
    img = Image.merge('YCbCr', colors).convert('RGB')
    """img.save('result/watermarkedImage.jpg')
    img.save('result/watermarkedImage.png') """
    return img


################ MAIN ################

# Paramètres
maxVal = 26 # 0-255 -> valeur dans l'image
DctCoef = [3,3] # 0-8 -> coefficient de la DCT à utiliser
password = "my_password" # Un string qui sera hashé

# Chemin des images en string :
image = "original/lena.jpg"
marque = "original/marque.png"
watermarkeImg = "watermarkedImage.jpg"

# 124 caractères max
texte = "Le tatouage numérique (en anglais digital watermark, « filigrane numérique ») est une technique permettant d'ajouter des inf"
#texte = "Test d'insertion de text  !"


### Fonctions d'insertion d'une image

watermarkeImage = embeddedImage(image,marque, password)
#watermarkeImage.save('watermarkedImage.jpg')

watermarkArray = recoverWatermark(watermarkeImg, password)
#watermarkArray.save('recoveredWatermark.png')


### Insertion d'un texte

watermarkeImage = embeddedTexte(image,texte, password)
#watermarkeImage.save('watermarkedImage.jpg')

watermarkArray = recoverText(watermarkeImg, password)
#print(watermarkArray)