from tkinter import *
from yahoo_fin import stock_info
import yfinance as yf
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
# Import the Required modules
#%%
# Inputting the name of the Stock and Storing it in a Variable
STK = input("Enter share name : ")

# Extract the Share information using the Ticker() Function
Share = yf.Ticker(STK).info

# Extracting the MarketPrice from the data
market_price = Share['regularMarketPrice']

# Printing the market price
print(market_price)

# This Code is Contrib


def stock_price():

    price = stock_info.get_live_price(e1.get())
    Current_stock.set(price)


master = Tk()
Current_stock = StringVar()

Label(master, text="Company Symbol : ").grid(row=0, sticky=W)
Label(master, text="Stock Result:").grid(row=3, sticky=W)

result2 = Label(master, text="", textvariable=Current_stock,
                ).grid(row=3, column=1, sticky=W)

e1 = Entry(master)
e1.grid(row=0, column=1)

b = Button(master, text="Show", command=stock_price)
b.grid(row=0, column=2, columnspan=2, rowspan=2, padx=5, pady=5)

mainloop()
