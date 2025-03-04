import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep
import argparse

hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath


# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
args = parser.parse_args()

# Load data and info
inData = np.load(args.inFilePath)
features = inData["features"]
info = inspectPath(os.path.dirname(args.inFilePath))

# get output directory
outDir = os.path.join(args.outDir if args.outDir else os.path.dirname(args.inFilePath), f"plots")
os.makedirs(outDir, exist_ok=True)
os.chmod(outDir, mode=0o777)

# plot config
pltConfig = {}
pltConfig["nelectron_asic_50perc"] = {
    "xlabel": r"V$_{\mathrm{TH}}$ Per Bit [mV]", 
    "ylabel": r"S-Curve Half Max [e$^{-}$]", 
    "idx" : 1,
    "fit" : "linear",
    "vthPerBit" : True,
    "legLoc" : "lower right"
}
pltConfig["scurve_mean"] = {
    "xlabel": r"V$_{\mathrm{TH}}$ Per Bit [mV]", 
    "ylabel": r"S-Curve $\mu$ [e$^{-}$]",
    "idx" : 2,
    "fit" : "linear",
    "vthPerBit" : True,
    "legLoc" : "lower right"
}
pltConfig["scurve_std"] = {
    "xlabel": r"V$_{\mathrm{TH}}$ Per Bit [mV]",
    "ylabel": r"S-Curve $\sigma$ [e$^{-}$]",
    "idx" : 3,
    "fit" : None,
    "vthPerBit" : True,
    "legLoc" : "upper right"
}

# Perform linear fit
def linear_func(x, a, b):
    return a * x + b

# VTH_{0-2} per bit from VTH
def vth_to_vthPerBit(vth, iB):
    scale = [50, 100, 200][iB]
    return vth * scale / 1250

def angleFromSlope(m, ax=None):
    if ax is None:
        aspect_ratio = 1
    else:
        # Get axis scaling factors
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Compute the scale-adjusted slope
        aspect_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    adjusted_slope = m / aspect_ratio
    # Compute the angle in degrees
    angle = np.arctan(adjusted_slope) * (180 / np.pi)
    return angle

# make plots
for name, config in pltConfig.items():

    # set up figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(config["xlabel"], fontsize=18, labelpad=10)
    ax.set_ylabel(config["ylabel"], fontsize=18, labelpad=10)

    # set y limit
    maxValue = np.max(features[:,:,config["idx"]])
    ylimMax = 1.35 * maxValue
    ax.set_ylim(0, ylimMax)

    # plot
    color = ["blue", "red", "orange"]
    for iB in range(3):
        # get vth per bit
        x_ = features[:,iB][:,0]
        if config["vthPerBit"]:
            x_ = vth_to_vthPerBit(x_, iB)
        # get y values
        y_ = features[:,iB][:,config["idx"]]
        mask = y_ > 0
        ax.plot(x_[mask], y_[mask], label=f'Bit {iB}', color=color[iB], marker='o', linestyle='-', markersize=4)
        
        # linear fit
        if config["fit"] == "linear":
            popt, pcov = curve_fit(linear_func, x_[mask], y_[mask])
            a, b = popt
            CvG = 1/a*1000000
            fit_label = f'y = {a:.2f}x {"-" if b < 0 else "+"} {abs(b):.2f}'
            fit_label = f'CvG = {CvG:.2f}uV/e-'
            ax.plot(x_[mask], linear_func(x_[mask], *popt), linestyle='--', color = color[iB], alpha=0.5)
            # Calculate the angle of the line for rotation
            if config["vthPerBit"]:
                x_text = 0.55
                y_text = 0.05*(iB+1)
                ax.text(x_text, y_text, fit_label, fontsize=12, color=color[iB], ha='left', va='bottom', transform=ax.transAxes, alpha=0.5)
            else:
                angle = 0 if config["vthPerBit"] else angleFromSlope(a, ax)
                x_text = x_[mask][0] + (x_[mask][-1] - x_[mask][0]) / 4
                y_text = linear_func(x_text, *popt)
                y_text += 0.1*y_text
                # print text label
                ax.text(x_text, y_text, fit_label, fontsize=12, color=color[iB], ha='left', va='bottom', rotation=angle, alpha=0.5)

    # make legend
    legend = ax.legend(fontsize=15, loc = "upper right") #bbox_to_anchor=(0.03, 0.85), loc='upper left')
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # set ticks
    SetTicks(ax)

    # add label and text
    SmartPixLabel(ax, 0.05, 0.9, size=22)
    ax.text(0.05, 0.85, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

    # save fig
    outFileName = os.path.join(outDir, f"MatrixVTH_{name}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()
