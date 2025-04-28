import numpy as np 
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep
import argparse

from SmartPixStyle import *
from Analyze import inspectPath

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
args = parser.parse_args()
# parent_directory = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2/2025.04.03_17.32.32_MatrixCalibration_vMin0.001_vMax0.400_vStep0.03400_nSample32.000_vdda0.900_VTH0.800_BXCLK10.00/"
parent_directory = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2/2025.04.17_10.36.11_MatrixCalibration_vMin0.001_vMax0.400_vStep0.03400_nSample32.000_vdda0.900_VTH0.800_BXCLK10.00/"
FileSetting = "settings.npy"
file_path = os.path.join(parent_directory, FileSetting)
# input file

inData = np.load(args.inFilePath)
features = inData["features"]
nelectron_asics = inData["nelectron_asics"]
scurve = inData["scurve"]
print("scurve shape = ", scurve.shape)
settings = np.load(file_path)
print(settings.shape)

# get file information
info = inspectPath(os.path.dirname(args.inFilePath))
print(info)

# get output directory
outDir = args.outDir if args.outDir else os.path.join(os.path.dirname(args.inFilePath), f"plots")
os.makedirs(outDir, exist_ok=True)

# mu_c is the pixel count per settings that respect the condition mu_bit2 > mu_bit1 > mu_bit0 > 0  
mu_c = np.zeros((features.shape[0]))
fifty_c = np.zeros((features.shape[0]))

print("setting shape=", (features.shape[0]))

for i in range(features.shape[0]):
    mu_c[i] = np.count_nonzero((features[i,:,2,2]>1.5*features[i,:,1,2]) & (features[i,:,1,2]>1.5*features[i,:,0,2]) & (features[i,:,0,2]>0))
    fifty_c[i] = np.count_nonzero((features[i,:,2,1]>features[i,:,1,1]) & (features[i,:,1,1]>features[i,:,0,1]) & (features[i,:,0,1]>0))

bestSettingResult = np.max(mu_c)
best_settings = np.argmax(mu_c)
top_10_indices = np.argsort(mu_c)[-10:][::-1]
nWorkingSetting = np.count_nonzero(mu_c>0)

print(nWorkingSetting)
# print(f"best setting index = {best_settings}, number of working pixel = {bestSettingResult}, setting = {settings[best_settings]}" )
# print(f"top 10 settings = {top_10_indices}" )
# print(f"top 10 settings = {mu_c[top_10_indices]}" )
print(f"mu bit2 for 5 pixels  = {features[best_settings,:,2,2][0:5]}" )
print(f"mu bit1 for 5 pixels  = {features[best_settings,:,1,2][0:5]}" )
print(f"mu bit0 for 5 pixels  = {features[best_settings,:,0,2][0:5]}" )
print(f"sigma bit2 for 5 pixels  = {features[best_settings,:,2,3][0:5]}" )
print(f"sigma bit2 for 5 pixels  = {features[best_settings,:,1,3][0:5]}" )
print(f"sigma bit2 for 5 pixels  = {features[best_settings,:,0,3][0:5]}" )

mu_c = np.array(mu_c)
print("mu_c = ", mu_c.shape) 
# mu_c = mu_c.reshape(40, 40)
print(settings.shape)
testSample = settings[:,4]
testDelay = settings[:,5]
print(testSample.shape)

# Create the plot
plt.scatter(testDelay, testSample, c=mu_c, cmap='viridis', s=256, edgecolor='black')  # Use a colormap

# Add labels
plt.xlabel('cfg_test_delay')
plt.ylabel('cfg_test_sample')
plt.title('2D Plot of working settings')

# Show the plot
plt.colorbar(label='Values')  # Display color bar to indicate values
# plt.show()

# save fig
outFileName = os.path.join(outDir, f"SCurve_ChipVersion{int(info['ChipVersion'])}_ChipID{int(info['ChipID'])}_SuperPixel{int(info['SuperPix'])}_nPix.pdf")
print(f"Saving file to {outFileName}")
plt.savefig(outFileName, bbox_inches='tight')
plt.close()



