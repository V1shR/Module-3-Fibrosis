import cv2
import numpy as np
import pandas as pd
from termcolor import colored

# 1. Configuration: Keep data organized together
data_input = {
    "filenames": [
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010067.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010107.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010087.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010019.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010051.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010104.jpg",
    ],
    "depths": [1500, 6300, 8000, 60, 400, 9700]
}

results = []

print(colored("Processing images...", "yellow"))

# 2. Single-pass loop: Load, Process, and Collect data in one go
for filepath, depth in zip(data_input["filenames"], data_input["depths"]):
    # Load image in grayscale and process immediately
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(colored(f"Error: Could not load {filepath}", "red"))
        continue

    # OpenCV countNonZero is faster than numpy sum for binary images
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    white_count = cv2.countNonZero(binary)
    total_pixels = binary.size
    black_count = total_pixels - white_count
    white_percent = (white_count / total_pixels) * 100

    # Store results in a list of dictionaries (optimal for Pandas)
    results.append({
        "Filenames": filepath,
        "Depths": depth,
        "White_Count": white_count,
        "Black_Count": black_count,
        "White percents": white_percent
    })

# 3. Output results (Printing and CSV)
print(colored("\nCounts of pixel by color in each image", "yellow"))
for i, res in enumerate(results):
    print(colored(f"White pixels in image {i}: {res['White_Count']}", "white"))
    print(colored(f"Black pixels in image {i}: {res['Black_Count']}", "grey")) # termcolor uses 'grey' for black-ish
    print(colored(f"{res['Filenames']}:", "red"))
    print(f"{res['White percents']:.4f}% White | Depth: {res['Depths']} microns\n")

# Create DataFrame and save
df = pd.DataFrame(results)
df[['Filenames', 'Depths', 'White percents']].to_csv('Percent_White_Pixels.csv', index=False)

print("The .csv file 'Percent_White_Pixels.csv' has been created.")

#Used Gemini for help in optimizing the code and making it more efficient.