import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from termcolor import colored
import time  # 1. Added time module

# --- Configuration ---
filenames = [
    r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010067.jpg",
    r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010107.jpg",
    r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010087.jpg",
    r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010019.jpg",
    r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010051.jpg",
    r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010104.jpg",
]

depths = [1500, 6300, 8000, 60, 400, 9700]

# --- Analysis Starts Here ---
print(colored("Starting analysis...", "yellow"))
start_time = time.time()  # 2. Start the clock

images = []
white_counts = []
black_counts = []
white_percents = []

# Build the list of all the images
for filename in filenames:
    img = cv2.imread(filename, 0)
    images.append(img)

# Calculate pixel counts
for x in range(len(filenames)):
    if images[x] is None:
        print(colored(f"Warning: Could not read image {x}", "red"))
        continue
        
    _, binary = cv2.threshold(images[x], 127, 255, cv2.THRESH_BINARY)
    white = np.sum(binary == 255)
    black = np.sum(binary == 0)
    white_counts.append(white)
    black_counts.append(black)

# Calculate percentages
for x in range(len(white_counts)):
    white_percent = (100 * (white_counts[x] / (black_counts[x] + white_counts[x])))
    white_percents.append(white_percent)

end_time = time.time()  # 3. Stop the clock
elapsed_time = end_time - start_time

# --- Results Output ---
print(colored("\nCounts of pixel by color in each image", "yellow"))
for x in range(len(white_percents)):
    print(colored(f"White pixels in image {x}: {white_counts[x]}", "white"))
    print(colored(f"Black pixels in image {x}: {black_counts[x]}", "grey")) # termcolor uses 'grey' for dark text
    print(colored(f'{filenames[x]}:', "red"))
    print(f'{white_percents[x]:.4f}% White | Depth: {depths[x]} microns\n')

# Create DataFrame and save
df = pd.DataFrame({
    'Filenames': filenames,
    'Depths': depths,
    'White percents': white_percents
})
df.to_csv('Percent_White_Pixels.csv', index=False)

print("The .csv file 'Percent_White_Pixels.csv' has been created.")

# 4. Display the total time taken
print(colored(f"Total time to analyze 6 images: {elapsed_time:.4f} seconds", "cyan"))