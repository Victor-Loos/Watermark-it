import inquirer
from watermark import *
from display import *
from attack import *


############## Parameters ##############

password = "my_password"

# Chemin des images
image = "original/cascade.jpg"
marque = "original/marque/dragon.jpg"
Iresult = "result/watermarkedImage.jpg"
Mresult = "result/recoveredWatermark.png"

# Texte à insérer
texte = "Le tatouage numérique (digital watermark, « filigrane numérique ») est une technique permettant d'ajouter des informations +++ les textes trop longs sont automatiquement coupés"
#texte = "Test d'insertion de text  !"



######### Version to use in terminal #########

def main():
    """ 
    Main function that asks the user if they want to 
    embed or extract a watermark and the format: image or text. 
    Then it asks for the password and the path of the image.
    """

    x = 1.3

    questions = [
        inquirer.List("type", message="Choose type", choices=["Embed", "Extract", "Display", "Attack"]),
        inquirer.List("format", message="Choose format", choices=["Image", "Text"])
    ]

    answers = inquirer.prompt(questions)
    watermark = None
    text = None

    if answers["type"] == "Embed":
        image_path = input("Enter path to original image: ")
        if answers["format"] == "Image":
            watermark_path = input("Enter path to watermark image: ")
            password = input("Choose a password: ")
            watermarked_image = embeddedImage(image_path, watermark_path, password)
            output_path = input("Enter path to save watermarked image: ")
            watermarked_image.save(output_path)
            print("Watermark embedded successfully in the image.")

        elif answers["format"] == "Text":
            text = input("Enter text to be used as watermark: ")
            password = input("Choose a password: ")
            watermarked_image = embeddedTexte(image_path, text, password)
            output_path = input("Enter path to save watermarked image: ")
            watermarked_image.save(output_path)
            print("Watermark embedded successfully in the image.")

    elif answers["type"] == "Extract":
        image_path = input("Enter path to watermarked image: ")
        if answers["format"] == "Image":
            password = input("Enter the password: ")
            extracted_watermark = recoverWatermark(image_path, password)
            output_path = input("Enter path to save extracted watermark: ")
            extracted_watermark.save(output_path)
            print("Watermark extracted successfully from the image.")

        elif answers["format"] == "Text":
            password = input("Enter the password: ")
            extracted_text = recoverText(image_path, password)
            print(f"Extracted text: {extracted_text}")


    elif answers["type"] == "Display":
        image = input("Enter path to original image: ")
        marque = input("Enter path to watermark image: ")
        Iresult = input("Enter path to watermarked image: ")
        Mresult = input("Enter path to extracted watermark: ")
        #x = int(input("Enter the size of the matrix used for the watermark: "))
        plotResult(image, marque, Iresult, Mresult, x)
        plotDiff(image, marque, Iresult, Mresult, x)        

    elif answers["type"] == "Attack":
        image = input("Enter path to original image: ")
        marque = input("Enter path to watermark image: ")
        Iresult = input("Enter path to watermarked image: ")
        Mresult = input("Enter path to extracted watermark: ")
        #x = int(input("Enter the size of the matrix used for the watermark: "))
        password = input("Enter the password: ")
        attackAll(image, marque, Iresult, Mresult, x, password)



if __name__ == "__main__":
    main()
