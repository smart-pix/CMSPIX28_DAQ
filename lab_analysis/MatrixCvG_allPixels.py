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
from Analyze import inspectPath, Cin, Qe, Pgain

# Perform linear fit
def linear_func(x, a, b):
    return a * x + b


# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
parser.add_argument('--staticPulse', action="store_true", help="Data from PreProgSCurveBurst_StaticPulse: fixed pulse amplitude (folder 'vth' label) with Vth swept (file 'vasic' label)")
parser.add_argument('--vthMin', type=float, default=0.03, help='Lower Vth edge [V] of the linear fit region')
args = parser.parse_args()

# Fit threshold [e-] = a * Vth [V] + b. Least squares assumes the scatter is in y, so regress
# on whichever variable was set by the routine: Vth in the normal scan, the injected charge
# in the static-pulse scan (there the measured Vth50 carries the pixel-to-pixel spread).
def fit_threshold_line(vth, q):
    if args.staticPulse:
        (m, c), _ = curve_fit(linear_func, q, vth)
        return 1 / m, -c / m
    (a, b), _ = curve_fit(linear_func, vth, q)
    return a, b


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
        # Analyze.py now leaves the S-curve axis in volts (nelectron_asics = v_asics / VtomV),
        # so the 50% point is a voltage and has to be converted to electrons here.
        if args.staticPulse:
            # In the static-pulse routine the roles are swapped: the folder "vth" value is the
            # injected pulse amplitude [V] and the S-curve axis is the swept Vth [V].
            # Swap back to (Vth [V], threshold [e-]).
            x, y = y, x * Pgain * Cin / Qe
        else:
            y = y * Pgain * Cin / Qe
        mask = y > 0
        linearRegion = x[mask] > args.vthMin
        # need at least 2 points inside the fit region for a straight line
        if np.count_nonzero(linearRegion) >= 2:
            try:
                a, b = fit_threshold_line(x[mask][linearRegion], y[mask][linearRegion])
                print(x[mask][linearRegion], y[mask][linearRegion])
                #a, b = popt
                #CvG = 1 / a * 1e6  # µV/e⁻
                #vth_offset = -b / a  # V
                #if CvG > 0:
                CvG = abs(1 / a) * 1e6   # µV/e⁻
                vth_offset = -b / a      # still the Vth where the fitted threshold hits 0 e⁻
                if np.isfinite(CvG):
                    bit_CvG[iB].append(CvG)
                    bit_VthOff[iB].append(vth_offset)  # V
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
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Fit failed for pixel {iP}, bit {iB}: {e}")
                continue

print("Pixels fitted per bit:", {iB: len(v) for iB, v in bit_CvG.items()})

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

    # Test the array to see if empty
    if y_all.size == 0:
        print(f"Skipped bit {iB})")
        continue


    # Perform a linear fit
    #popt, _ = curve_fit(linear_func, x_all, y_all)
    #a, b = popt
    a, b = fit_threshold_line(x_all, y_all)

    # Generate fitted line
    x_fit = np.linspace(min(x_all), max(x_all), 100)
    y_fit = linear_func(x_fit, a, b)

    # Plot the data and the fit
    plt.figure()
    plt.scatter(x_all, y_all, label='Data', color='blue', alpha=0.6, s=10)    
    plt.plot(x_fit, y_fit, label=f'Fit: y = {a:.2f}x + {b:.2f}', color='red')
    plt.xlim(0.03, 0.33)
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

# ====== CvG / Vth offset Histogram Plotting ======
# per-bit histograms; an empty bit is labelled instead of silently left blank
def plotPerBit(bitVals, xlabel, titleFmt, outName):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for iB in range(nBits):
        vals = np.array(bitVals[iB])
        axs[iB].set_xlabel(xlabel)
        axs[iB].set_ylabel("Count")
        axs[iB].grid(True)
        if len(vals) == 0:
            axs[iB].set_title(f'Bit {iB}: no pixels passed')
            continue
        mu, std = norm.fit(vals)
        axs[iB].hist(vals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axs[iB].set_title(titleFmt.format(iB=iB, mu=mu, std=std))
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, outName))
    plt.close()

# all bits combined; skipped when empty (norm.fit of an empty array returns nan)
def plotCombined(bitVals, xlabel, titleFmt, outName):
    vals = np.concatenate([np.array(v) for v in bitVals.values()])
    if len(vals) == 0:
        print(f"No fitted pixels, skipping {outName}")
        return
    mu, std = norm.fit(vals)
    plt.figure(figsize=(8,6))
    plt.hist(vals, bins=40, color='salmon', edgecolor='black', alpha=0.75)
    plt.title(titleFmt.format(mu=mu, std=std))
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(outDir, outName))
    plt.close()

plotPerBit(bit_CvG, "CvG [µV/e⁻]", 'Bit {iB}: μ = {mu:.2f} µV/e⁻, σ = {std:.2f}', "CvG_Histograms_PerBit.pdf")
plotPerBit(bit_VthOff, "Vth offset [V]", 'Bit {iB}: Vth offset = {mu:.3f} V, σ = {std:.3f}', "vthOffset_Histograms_PerBit.pdf")
plotCombined(bit_CvG, "CvG [µV/e⁻]", 'All Bits Combined: μ = {mu:.2f} µV/e⁻, σ = {std:.2f}', "CvG_Histogram_Combined.pdf")
plotCombined(bit_VthOff, "Vth offset [V]", 'All Bits Combined: μ = {mu:.3f} V, σ = {std:.3f}', "vth_offset_Histogram_Combined.pdf")

# Save CvG data as (iP, iB, CvG)
CvG_array = np.array(store_CvG)
vth_offset_array = np.array(store_VthOff)
save_path1 = os.path.join(outDir, "CvG_data.npy")
save_path2 = os.path.join(outDir, "vth_offset_data.npy")
np.save(save_path1, CvG_array)
np.save(save_path2, vth_offset_array)
print(f"Saved CvG data to: {save_path1}")
print(f"Saved vth offset data to: {save_path2}")
