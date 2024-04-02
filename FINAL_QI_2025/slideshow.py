import os
from itertools import cycle
from tkinter import Tk, Label
from PIL import Image, ImageTk


class Slideshow:
    def __init__(self, parent):
        # Directory containing images
        self.image_dir = "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/charts"

        # Get all image files in the directory
        self.image_files = [os.path.join(self.image_dir, f) for f in os.listdir(
            self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # Cycle through the images (loop back to the start when reaching the end)
        self.image_cycle = cycle(self.image_files)

        # Create a label to display the images
        self.image_label = Label(parent)
        self.image_label.pack()

        # Start the slideshow
        self.update_image()

    def update_image(self):
        # Get the next image file path
        image_path = next(self.image_cycle)

        # Open the image file and convert it to a format Tkinter can handle
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        # Update the label with the new image
        self.image_label.config(image=photo)
        self.image_label.image = photo

        # Set a timer to update the image after a delay (e.g., 3000 milliseconds)
        self.image_label.after(3000, self.update_image)


# Create the root window
root = Tk()

# Create an instance of Slideshow and pass the root window to it
slideshow_instance = Slideshow(root)

# Run the Tkinter event loop
root.mainloop()
