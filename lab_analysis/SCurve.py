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
print(scurve.shape)

# get file information
info = inspectPath(os.path.dirname(args.inFilePath))
print(info)

# get output directory
outDir = args.outDir if args.outDir else os.path.dirname(args.inFilePath)
os.makedirs(outDir, exist_ok=True)
# os.chmod(outDir, mode=0o777)

# Define a base color palette for each bit
bit_colors = ['red', 'green', 'blue']
base_colors = {
    0: plt.cm.Reds(np.linspace(0.1, 1, scurve.shape[1])),
    1: plt.cm.Greens(np.linspace(0.1, 1, scurve.shape[1])),
    2: plt.cm.Blues(np.linspace(0.1, 1, scurve.shape[1]))
}
print(base_colors, scurve.shape)

# loop over the settings
for iS in range(scurve.shape[0]):

    # set up figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel("Electrons", fontsize=18, labelpad=10)
    ax.set_ylabel("Fraction of Samples", fontsize=18, labelpad=10)
    ax.set_ylim(0, 1.24)

    # loop over the pixels
    for iP in range(scurve.shape[1]):

        # loop over the bits
        for iB in range(scurve.shape[2]):
            # settings
            color = None
            label=f"Bit {iB}" if iP == (scurve.shape[1]-1) else None
            if info["testType"] == "Single":
                color = bit_colors[iB]
            elif info["testType"] in ["MatrixNPix", "MatrixCalibration"]:
                color = base_colors[iB][iP]
            elif info["testType"] == "MatrixVTH":
                label = ""
            # plot
            # print(iS, iP, iB, scurve[iS, iP, iB])
            ax.plot(nelectron_asics, scurve[iS, iP, iB], color=color, label=label, marker='o', ms=5)
            # print(f"Pixel {iP}, Bit {iB} (50%, mu, sigma): ", features[iS, iP, iB][1:])

    # add legend
    legend = ax.legend(fontsize=12, loc='upper right', ncol=1, bbox_to_anchor=(0.95, 1.01))
    texts = legend.get_texts()
    for text, color in zip(texts, bit_colors):
        text.set_fontweight('bold')
        legend.legend_handles[texts.index(text)].set_color(color)

    # set ticks
    SetTicks(ax)

    # add label and text
    SmartPixLabel(ax, 0.05, 0.9, size=22)
    ax.text(0.05, 0.85, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

    # save fig
    outFileName = os.path.join(outDir, f"SCurve_ChipVersion{int(info['ChipVersion'])}_ChipID{int(info['ChipID'])}_SuperPixel{int(info['SuperPix'])}_Setting{iS}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()
