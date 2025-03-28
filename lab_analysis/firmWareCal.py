import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')

args = parser.parse_args()

# input file

# Define the x and y coordinates
settings = np.load(os.path.join(args.inFilePath, "settingList.npy"))
x = settings[0]
y = settings[1]
            
# Define the values you want to plot (array [0 0 0 0 0 0 1 1 1])
values = np.load(os.path.join(args.inFilePath, "settingPass.npy"))

# Create the plot
plt.scatter(x, y, c=values, cmap='viridis', s=100, edgecolor='black')  # Use a colormap

# Add labels
plt.xlabel('cfg_test_delay')
plt.ylabel('cfg_test_sample')
plt.title('2D Plot of working settings')

# Show the plot
plt.colorbar(label='Values')  # Display color bar to indicate values
plt.show()
