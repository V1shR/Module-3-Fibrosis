import cv2
import numpy as np
import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# 1. Configuration: Keep data organized together
data_input = {
    "filenames": [
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010067.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010107.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010087.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010019.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010051.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010104.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010103.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010068.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010118.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010098.jpg",
        r"/Users/vasishtramineni/Downloads/Computational BME/Module-3-Fibrosis/images/MASK_SK658 Slobe ch010136.jpg"
        

    ],
    "depths": [1500, 6300, 8000, 60, 400, 9700,9600,9800,9900,10000,9200]
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

##############
# LECTURE 2: UNCOMMENT BELOW

# Interpolate a point: given a depth, find the corresponding white pixel percentage

# 1. Pull the full lists from your DataFrame
x = df['Depths'].values
y = df['White percents'].values

# 2. Get user input
interpolate_depth = float(input(colored(
    "Enter the depth at which you want to interpolate a point (in microns): ", "yellow")))

# 3. Create the interpolation function using the full datasets
# Note: x must be sorted for interp1d to work reliably
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]

i = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate")
interpolate_point = float(i(interpolate_depth))

print(colored(
    f'The interpolated point is at the x-coordinate {interpolate_depth} and y-coordinate {interpolate_point:.4f}.', "green"))

# 4. Prepare data for plotting
# Create copies of the original lists to add the new point
depths_i = list(x)
white_percents_i = list(y)

depths_i.append(interpolate_depth)
white_percents_i.append(interpolate_point)

# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Plot 1: Original Data
axs[0].scatter(x, y, color='blue')
axs[0].set_title('Depth vs White Pixel Percentage')
axs[0].set_xlabel('Depth (microns)')
axs[0].set_ylabel('% White Pixels')
axs[0].grid(True)

# Plot 2: Data + Interpolated Point
axs[1].scatter(x, y, color='blue', label='Original Data')
axs[1].scatter(interpolate_depth, interpolate_point, color='red', s=100, label='Interpolated Point', zorder=5)
axs[1].set_title('Depth vs White Pixel Percentage (with Interpolated Point)')
axs[1].set_xlabel('Depth (microns)')
axs[1].set_ylabel('% White Pixels')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()