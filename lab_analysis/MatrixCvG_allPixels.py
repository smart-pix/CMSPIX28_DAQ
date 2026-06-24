import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep
import argparse
import json

# import MatrixVTH.linear_func as linear_func
from scipy.stats import norm

hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath

# Perform linear fit
def linear_func(x, a, b):
    return a * x + b


# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
args = parser.parse_args()

# Load data and info
inData = np.load(args.inFilePath)
features = inData["features"]

# get information
info = inspectPath(os.path.dirname(args.inFilePath))
print(info)

# get output directory
outDir = args.outDir if args.outDir else os.path.dirname(args.inFilePath)
os.makedirs(outDir, exist_ok=True)
# os.chmod(outDir, mode=0o777)
print("Computing CvG for all pixels and bit across all settings...")
store_CvG = []
store_VthOff = []

bit_VthOff = {0: [], 1: [], 2: []}
bit_CvG = {0: [], 1: [], 2: []}

nSettings, nPixels, nBits, nFeatures = features.shape

# Initialize a list to store the data
fit_data = []

for iB in range(nBits):
    for iP in range(nPixels):
        x = []
        y = []
        for iS in range(nSettings):
            vth = features[iS, iP, iB, 1]
            value = features[iS, iP, iB, 2]  # 50% electron value
            if value > 0 and vth > 0:
                x.append(vth)
                y.append(value)
        x = np.array(x)
        y = np.array(y)
        if len(x) >= 2:
            try:
                mask = y > 0
                linearRegion = x[mask] > 0.03 # 0.05
                popt, pcov = curve_fit(linear_func, x[mask][linearRegion], y[mask][linearRegion])
                print(x[mask][linearRegion], y[mask][linearRegion])
                a, b = popt
                CvG = 1 / a * 1e6  # µV/e⁻
                vth_offset = -b / a  # V
                if CvG > 0:
                    bit_CvG[iB].append(CvG)
                    bit_VthOff[iB].append(vth_offset)  # mV
                    store_CvG.append((iP, iB, CvG))
                    store_VthOff.append((iP, iB, vth_offset))

                    # Save the data for this pixel and bit
                    fit_data.append({
                        "pixel": iP,
                        "bit": iB,
                        "x": x[mask][linearRegion].tolist(),
                        "y": y[mask][linearRegion].tolist(),
                        "CvG": CvG,
                        "vth_offset": vth_offset
                    })
            except RuntimeError:
                continue

# Save the fit data to a JSON file
with open(os.path.join(outDir, f'fit_data.json'), 'w') as json_file:
    json.dump(fit_data, json_file, indent=4)


with open(os.path.join(outDir, f'fit_data.json'), 'r') as json_file:
    fit_data = json.load(json_file)

# Iterate over each bit and plot the data
for iB in range(nBits):
    x_all = []
    y_all = []

    # Collect x and y data for all pixels for the current bit
    for entry in fit_data:
        if entry['bit'] == iB:
            x_all.extend(entry['x'])
            y_all.extend(entry['y'])

    # Convert to numpy arrays
    x_all = np.array(x_all)
    y_all = np.array(y_all)

    # Perform a linear fit
    popt, _ = curve_fit(linear_func, x_all, y_all)
    a, b = popt

    # Generate fitted line
    x_fit = np.linspace(min(x_all), max(x_all), 100)
    y_fit = linear_func(x_fit, a, b)

    # Plot the data and the fit
    plt.figure()
    plt.scatter(x_all, y_all, label='Data', color='blue', alpha=0.6, s=10)    
    plt.plot(x_fit, y_fit, label=f'Fit: y = {a:.2f}x + {b:.2f}', color='red')
    plt.xlim(0.02, 0.09)
    plt.title(f'Bit {iB}')
    plt.xlabel('Vth [V]')
    plt.ylabel('S-curve half max [e-]')
    plt.legend()
    plt.grid()
    # Display fit parameters on the plot with a border at the bottom-right
    plt.text(0.95, 0.05, f'Threshold [e-] = {a:.2f} * V$_{{TH}}$ [V] + {b:.2f}', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(edgecolor='black', facecolor='none', boxstyle='round,pad=0.5'))
    plt.savefig(os.path.join(outDir, f'CvG_AllPixelsFit_Bit_{iB}.png'))

# # ====== CvG Histogram Plotting ======
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# for iB in range(nBits):
#     vals = np.array(bit_CvG[iB])
#     if len(vals) == 0:
#         continue
#     mu, std = norm.fit(vals)
#     axs[iB].hist(vals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#     axs[iB].set_title(f'Bit {iB}: μ = {mu:.2f} µV/e⁻, σ = {std:.2f}')
#     axs[iB].set_xlabel("CvG [µV/e⁻]")
#     axs[iB].set_ylabel("Count")
#     axs[iB].grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(outDir, "CvG_Histograms_PerBit.pdf"))
# plt.close()

# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# for iB in range(nBits):
#     vals = np.array(bit_VthOff[iB])
#     if len(vals) == 0:
#         continue
#     mu, std = norm.fit(vals)
#     axs[iB].hist(vals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#     axs[iB].set_title(f'Bit {iB}: Vth offset = {mu:.2f} V, σ = {std:.2f}')
#     axs[iB].set_xlabel("Vth offset [mV]")
#     axs[iB].set_ylabel("Count")
#     axs[iB].grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(outDir, "vthOffset_Histograms_PerBit.pdf"))
# plt.close()

# # Combined histogram
# combined_vals = np.concatenate([np.array(v) for v in bit_CvG.values()])
# mu, std = norm.fit(combined_vals)
# plt.figure(figsize=(8,6))
# plt.hist(combined_vals, bins=40, color='salmon', edgecolor='black', alpha=0.75)
# plt.title(f'All Bits Combined: μ = {mu:.2f} µV/e⁻, σ = {std:.2f}')
# plt.xlabel("CvG [µV/e⁻]")
# plt.ylabel("Count")
# plt.grid(True)
# plt.savefig(os.path.join(outDir, "CvG_Histogram_Combined.pdf"))
# plt.close()

# # Combined histogram
# combined_vals = np.concatenate([np.array(v) for v in bit_VthOff.values()])
# mu, std = norm.fit(combined_vals)
# plt.figure(figsize=(8,6))
# plt.hist(combined_vals, bins=40, color='salmon', edgecolor='black', alpha=0.75)
# plt.title(f'All Bits Combined: μ = {mu:.2f} V⁻, σ = {std:.2f}')
# plt.xlabel("vth offset [mV]")
# plt.ylabel("Count")
# plt.grid(True)
# plt.savefig(os.path.join(outDir, "vth_offset_Histogram_Combined.pdf"))
# plt.close()

# # Save CvG data as (iP, iB, CvG)
# CvG_array = np.array(store_CvG)
# vth_offset_array = np.array(store_VthOff)
# save_path1 = os.path.join(outDir, "CvG_data.npy")
# save_path2 = os.path.join(outDir, "vth_offset_data.npy")
# np.save(save_path1, CvG_array)
# np.save(save_path2, vth_offset_array)
# print(f"Saved CvG data to: {save_path1}")
# print(f"Saved vth offset data to: {save_path2}")
