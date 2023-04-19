import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import customtkinter
import configparser
import subprocess, sys

from watermark import *
from display import *
from attack import *

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        # configure window
        self.title("Watermark-it")
        #self.geometry("800x640")
        # Min size of the window
        self.minsize(950, 550)
        # Fix the size of the window
        self.resizable(False, False)
        # Center the tkinter window on the screen and adapt to window size
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = 950
        window_height = 550
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        
        ############# Initialize the menu bar 
        
        # Create a canvas to display the default image
        self.image_canvas = tk.Canvas(self, width=300, height=300)
        self.image_canvas.grid(row=1, column=0, padx=20, pady=20) # padx for horizontal padding, pady for vertical padding
        self.image_label = customtkinter.CTkLabel(master=self, text="Image principale", font=("Helvetica", 17))
        # Add padding to the label only on top
        self.image_label.grid(row=0, column=0, pady=(20, 0))

        
        # Create a button to select a new image
        self.button = customtkinter.CTkButton(master=self, text="Select Image", command=self.load_image)
        self.button.grid(row=2, column=0, padx=10, pady=10)
        
        # Create a button to open the settings window
        self.button = customtkinter.CTkButton(master=self, text="Settings")#, command=self.open_settings)
        self.button.grid(row=2, column=1, padx=10, pady=10)
        self.button.configure(state="disabled")
        
        # Create a button to select a new image for the second canvas
        self.button = customtkinter.CTkButton(master=self, text="Select Mark", command=self.load_mark)
        self.button.grid(row=2, column=2, padx=10, pady=10)
        
        

        # Create a frame to display all the buttons
        self.main_frame = customtkinter.CTkFrame(master=self, width=300, height=300)
        self.main_frame.grid(row=1, column=1, padx=20, pady=20)
        
        # Add a textbox in the frame to input the password
        self.password_label = customtkinter.CTkLabel(master=self.main_frame, text="Mot de passe")
        self.password_label.grid(row=0, column=0, padx=20, pady=(20, 0))
        self.password_entry = customtkinter.CTkEntry(master=self.main_frame, width=200, height=20, placeholder_text="Mot de passe")
        self.password_entry.grid(row=1, column=0, padx=20, pady=(5, 20))
        
        # Add a button to instert the mark
        self.insert_button = customtkinter.CTkButton(master=self.main_frame, text="Insérer", command=self.insert_mark)
        self.insert_button.grid(row=2, column=0, padx=20, pady=20)
        
        # Add a button to extract the mark
        self.extract_button = customtkinter.CTkButton(master=self.main_frame, text="Extraire", command=self.extract_mark)
        self.extract_button.grid(row=3, column=0, padx=20, pady=20)
        
        # Add a button to show the image in the file explorer
        self.show_button = customtkinter.CTkButton(master=self.main_frame, text="Taille maximum ?", command=self.get_size)
        self.show_button.grid(row=4, column=0, padx=20, pady=20)
        


        # Create a canvas to display the mark image inside a tabview
        self.mark_label = customtkinter.CTkLabel(master=self, text="Marque", font=("Helvetica", 17))
        self.mark_label.grid(row=0, column=2, pady=(20, 0))
        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=260)
        self.tabview.grid(row=1, column=2, padx=20, pady=(0, 20), sticky="nsew")
        self.tabview.add("Image")
        self.tabview.add("Texte")
        
        self.tabview.tab("Image").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Texte").grid_columnconfigure(0, weight=1)

        self.mark_canvas = customtkinter.CTkCanvas(self.tabview.tab("Image"), width=260, height=260)
        self.mark_canvas.grid(row=0, column=0, padx=0, pady=0)
        self.textbox = customtkinter.CTkTextbox(self.tabview.tab("Texte"), width=260, height=260)
        self.textbox.grid(row=0, column=0, padx=0, pady=0)
        

        self.progressbar = customtkinter.CTkProgressBar(master=self)
        self.progressbar.grid(row=5, column=0, columnspan=3, padx=20, pady=20, sticky="ew")
        self.progressbar.set(1)


        # Add a button for comparaison
        self.compare_button = customtkinter.CTkButton(master=self, text="Comparer les images", command=self.compare_images)
        self.compare_button.grid(row=6, column=0, padx=10, pady=10)
        
        # Add a button to attack the image
        self.attack_button = customtkinter.CTkButton(master=self, text="Attaquer l'image", command=self.attack_image)
        self.attack_button.grid(row=6, column=2, padx=10, pady=10)


        
        ############# Load the last image file path from the configuration file
        
        # Load the last image file path from the configuration file
        self.config = configparser.ConfigParser()
        # Check if the configuration file exists
        if os.path.exists("config.ini"):
            self.config.read("config.ini")
        # Check if the General section exists in the configuration file
        if self.config.has_section("General"):
            self.last_image_path = self.config.get("General", "last_image_path", fallback=None)
        else:
            self.last_image_path = None
        # If there is a last image file path, load it
        if self.last_image_path is not None:
            self.load_image(self.last_image_path)
                    
        # Load the last mark file path from the configuration file
        self.config = configparser.ConfigParser()
        if os.path.exists("config.ini"):
            self.config.read("config.ini")
        if self.config.has_section("General"):
            self.last_mark_path = self.config.get("General", "last_mark_path", fallback=None)
        else:
            self.last_mark_path = None
        if self.last_mark_path is not None:
            self.load_mark(self.last_mark_path)
            
    
            
            

        ############ Load the default image
        
        # Load the default image
        # If an image is not selected, display a gray rectangle
        if not hasattr(self, "image"):
            self.image_canvas.create_rectangle(0, 0, 300, 300, fill="gray")
            
        # Load the default mark
        # If an image is not selected, display a gray rectangle
        if not hasattr(self, "mark"):
            self.mark_canvas.create_rectangle(0, 0, 260, 260, fill="gray")
            
        


        
    def load_image(self, file_path=None):
        # If a file path is not provided, prompt the user to select an image file
        if file_path is None:
            file_path = filedialog.askopenfilename()

        # Load the selected image file
        try:
            self.image = Image.open(file_path)
        except FileNotFoundError:
            return
        
        # Resize the image to fit the canvas
        self.image = self.image.resize((300, 300), Image.ANTIALIAS)

        # Update the canvas with the new image
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Save the last image file path to the configuration file
        self.last_image_path = file_path
        if not self.config.has_section("General"):
            self.config.add_section("General")
        self.config.set("General", "last_image_path", file_path)
        with open("config.ini", "w") as config_file:
            self.config.write(config_file)
            
            
    def load_mark(self, file_path=None):
        # If a file path is not provided, prompt the user to select an image file
        if file_path is None:
            file_path = filedialog.askopenfilename()

        # Load the selected image file
        try:
            self.mark = Image.open(file_path)
        except FileNotFoundError:
            return
        
        # Resize the image to fit the canvas
        self.mark = self.mark.resize((260, 260), Image.ANTIALIAS)
        
        # Update the canvas with the new image
        self.photo2 = ImageTk.PhotoImage(self.mark)
        self.mark_canvas.create_image(0, 0, image=self.photo2, anchor=tk.NW)
        
        # Save the last image file path to the configuration file
        self.last_mark_path = file_path
        if not self.config.has_section("General"):
            self.config.add_section("General")
        self.config.set("General", "last_mark_path", file_path)
        with open("config.ini", "w") as config_file:
            self.config.write(config_file)
            
    def load_wimg(self, file_path=None):
        # If a file path is not provided, prompt the user to select an image file
        if file_path is None:
            file_path = filedialog.askopenfilename()

        # Load the selected image file
        try:
            self.wimg = Image.open(file_path)
        except FileNotFoundError:
            return
        
        # Resize the image to fit the canvas
        self.wimg = self.wimg.resize((300, 300), Image.ANTIALIAS)
        
        # Update the canvas with the new image
        self.photo3 = ImageTk.PhotoImage(self.wimg)
        self.wimg_canvas.create_image(0, 0, image=self.photo3, anchor=tk.NW)
        
        # Save the last image file path to the configuration file
        self.last_wimg_path = file_path
        if not self.config.has_section("General"):
            self.config.add_section("General")
        self.config.set("General", "last_wimg_path", file_path)
        with open("config.ini", "w") as config_file:
            self.config.write(config_file)
        

    def insert_mark(self):
        # Start the progress bar
        self.progressbar.start()
        
        # Get the password from the password entry if it is not empty
        if self.password_entry.get() != "":
            password = self.password_entry.get()
        else:
            password = None
            
        # Get the tabview name currently selected
        tabview = self.tabview.get()
        
        # If the tabview is the first tab, then the user wants to watermark an image
        if tabview == "Image":
            # Pass the image, the mark path and password to the embeddedImage function
            watermarkeImage = embeddedImage(self.last_image_path, self.last_mark_path, password)
            
            # Save the watermarked image in system dowlnoad folder with the 'bicubic' interpolation
            #watermarkeImage.save(os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg'), interpolation="bicubic")
            # Save the watermarked image in system dowlnoad folder with jpg quality 80 (default is 75)
            watermarkeImage.save(os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg'))#, quality=80)
            
            # Show the watermarked image in the file system
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg')])
                
        elif tabview == "Texte":
            # Pass the image, the mark path and password to the embeddedTexte function
            watermarkeTexte = embeddedTexte(self.last_image_path, self.textbox.get("0.0", "end"), password)
            
            # Save the watermarked image in system dowlnoad folder
            watermarkeTexte.save(os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg'))
                        # Show the watermarked image in the file system
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg')])
            
        # Save the watermarked image in the configuration file
        self.last_wimg_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg')
        if not self.config.has_section("General"):
            self.config.add_section("General")
        self.config.set("General", "last_wimg_path", os.path.join(os.path.expanduser('~'), 'Downloads', 'watermarked_image.jpg'))
        with open("config.ini", "w") as config_file:
            self.config.write(config_file)
            
        # Stop the progress bar
        self.progressbar.stop()
        
            
    def extract_mark(self):
        # Start the progress bar
        self.progressbar.start()
        
        # Get the password from the password entry if it is not empty
        if self.password_entry.get() != "":
            password = self.password_entry.get()
        else:
            password = None
        
        # Ask the user if he wants to use the last watermarked image path or the last image path
        if messagebox.askyesno("Choix de l'image", "Voulez-vous utiliser la dernière image filigranée ?"):
            # Read the last watermarked image path from the configuration file
            self.last_wimg_path = self.config.get("General", "last_wimg_path")
        else:
            # Read the last image path from the configuration file
            self.last_wimg_path = self.config.get("General", "last_image_path")
        
        # Get the tabview name currently selected
        tabview = self.tabview.get()
        
        # If the tabview is the first tab, then the user wants to watermark an image
        if tabview == "Image":
            # Pass the image path and password to the recoverWatermark function
            watermarkArray = recoverWatermark(self.last_wimg_path, password)
            # Save the extracted mark image in system dowlnoad folder
            watermarkArray.save(os.path.join(os.path.expanduser('~'), 'Downloads', 'extracted_mark.jpg'))
            # Show the extracted mark image in the file system
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, os.path.join(os.path.expanduser('~'), 'Downloads', 'extracted_mark.jpg')])
            
            # Save the extracted mark image in the configuration file
            self.last_emark_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'extracted_mark.jpg')
            if not self.config.has_section("General"):
                self.config.add_section("General")
            self.config.set("General", "last_emark_path", os.path.join(os.path.expanduser('~'), 'Downloads', 'extracted_mark.jpg'))
            with open("config.ini", "w") as config_file:
                self.config.write(config_file)
                
        elif tabview == "Texte":
            # Pass the image path and password to the recoverText function
            watermarkText = recoverText(self.last_wimg_path, password)
            # Show the extracted mark text in a messagebox
            #messagebox.showinfo("Texte extrait", "Le texte extrait est : \n" + watermarkText)
            # Save the extracted mark text in the file system
            with open(os.path.join(os.path.expanduser('~'), 'Downloads', 'extracted_mark.txt'), "w") as text_file:
                text_file.write(watermarkText)
            # Show the extracted mark text in the file system
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, os.path.join(os.path.expanduser('~'), 'Downloads', 'extracted_mark.txt')])
            
        # Stop the progress bar
        self.progressbar.stop()
        
    def get_size(self):
        imgMark, txtMark = getMarkSize(self.last_image_path)
        # Show the size of the image mark
        messagebox.showinfo("Taille du filigrane", "La taille d'insertion du filigrane est : \nEn image : " + str(imgMark) + "\nEn texte : " + str(txtMark) + " caractères")
        
    def compare_images(self):
        self.progressbar.start()
        x = 1.3
        image = self.last_image_path
        marque = self.last_mark_path
        Iresult = self.last_wimg_path
        Mresult = self.last_emark_path
        plotResult(image, marque, Iresult, Mresult, x)
        plotDiff(image, marque, Iresult, Mresult, x) 
        self.progressbar.stop()
        
    def attack_image(self):
        self.progressbar.start()
        x = 1.3
        # Get the password from the password entry if it is not empty
        if self.password_entry.get() != "":
            password = self.password_entry.get()
        else:
            password = None
        image = self.last_image_path
        Iresult = self.last_wimg_path
        
        tabview = self.tabview.get()
        if tabview == "Image":
            marque = self.last_mark_path
            Mresult = self.last_emark_path
                
            attackAll(image, marque, Iresult, Mresult, x, password)
        elif tabview == "Texte":
            marque = self.textbox.get("0.0", "end")

            attackAllText(image, marque, Iresult, password)
        
        self.progressbar.stop()


if __name__ == "__main__":
    app = App()
    app.mainloop()