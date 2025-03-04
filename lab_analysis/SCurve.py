import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep
import argparse

# plt.style.use(hep.style.ROOT)
hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
args = parser.parse_args()

# input file
inData = np.load(args.inFilePath)
features = inData["features"]
nelectron_asics = inData["nelectron_asics"]
scurve = inData["scurve"]

# get file information
info = inspectPath(os.path.dirname(args.inFilePath))

# get output directory
outDir = os.path.join(args.outDir if args.outDir else os.path.dirname(args.inFilePath), f"plots")
os.makedirs(outDir, exist_ok=True)
os.chmod(outDir, mode=0o777)

# set up figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel("Electrons", fontsize=18, labelpad=10)
ax.set_ylabel("Fraction of Samples", fontsize=18, labelpad=10)
ax.set_ylim(0, 1.24)

# plot all bits on the same plot
for iP in range(scurve.shape[0]):
    for iB in range(scurve.shape[1]):
        ax.plot(nelectron_asics, scurve[iP, iB])
        print(f"Pixel {iP}, Bit {iB} (50%, mu, sigma): ", features[iP, iB][1:])

# set ticks
SetTicks(ax)

# add label and text
SmartPixLabel(ax, 0.05, 0.9, size=22)
ax.text(0.05, 0.85, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

# save fig
outFileName = os.path.join(outDir, f"SCurve_ChipVersion{int(info['ChipVersion'])}_ChipID{int(info['ChipID'])}_SuperPixel{int(info['SuperPix'])}.pdf")
print(f"Saving file to {outFileName}")
plt.savefig(outFileName, bbox_inches='tight')
plt.close()