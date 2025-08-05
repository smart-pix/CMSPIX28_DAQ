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

# get information
info = inspectPath(os.path.dirname(args.inFilePath))
print(info)

# get output directory
outDir = args.outDir if args.outDir else os.path.dirname(args.inFilePath)
os.makedirs(outDir, exist_ok=True)
# os.chmod(outDir, mode=0o777)

# plot config
xlabel = "Injection Delay [ns]" #"Injection Delay [2.5 ns steps]"
pltConfig = {
    "nelectron_asic_50perc" : {
        "xlabel": xlabel,
        "ylabel": r"S-Curve Half Max [e$^{-}$]", 
        "idx" : 1,
        "legLoc" : "lower left"
    },
    "scurve_mean" : {
        "xlabel": xlabel,
        "ylabel": r"S-Curve $\mu$ [e$^{-}$]",
        "idx" : 2,
        "legLoc" : "lower left"
    },
    "scurve_std" : {
        "xlabel": xlabel,
        "ylabel": r"S-Curve $\sigma$ [e$^{-}$]",
        "idx" : 3,
        "legLoc" : "lower left"
    }
}

# make plots
for name, config in pltConfig.items():
    
    # no good values then skip
    if np.all(features[:,:,:,config["idx"]] == -999):
        print(f"All values are -999 for feature {config['idx']}. Skipping plotting.")
        continue

    # plot
    color = ["blue", "red", "orange"]
    for iS in range(features.shape[0]):
        
        # set up figure
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlabel(config["xlabel"], fontsize=18, labelpad=10)
        ax.set_ylabel(config["ylabel"], fontsize=18, labelpad=10)
        
        # # set y limit
        # maxValue = np.max(features[:,:,:,config["idx"]])
        # ylimMax = 1.35 * maxValue
        # ax.set_ylim(0, ylimMax)

        # save limits
        xlimMax, ylimMax = 0, 0
        for iB in range(3):
            # get vth per bit
            x_ = features[iS:,:,iB,0].flatten()
            # get y values
            y_ = features[iS:,:,iB,config["idx"]].flatten()
            mask = y_ > 0
            # plot
            ax.plot(x_[mask], y_[mask], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=4)
            # save maxes
            if np.max(x_[mask]) > xlimMax:
                xlimMax = np.max(x_[mask])
            if np.max(y_[mask]) > ylimMax:
                ylimMax = np.max(y_[mask])

        # set limits
        ax.set_xlim(0, xlimMax * 1.1)
        ax.set_ylim(0, ylimMax * 1.1)

        # make legend
        legend = ax.legend(fontsize=15, loc = config["legLoc"]) #bbox_to_anchor=(0.03, 0.85), loc='upper left')
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # set ticks
        SetTicks(ax)

        # add label and text
        SmartPixLabel(ax, 0.05, 0.9, size=22)
        ax.text(0.05, 0.85, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
        ax.text(0.05, 0.80, f"Pixel {int(info['nPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

        # save fig
        outFileName = os.path.join(outDir, f"MatrixBxCLKDly_{name}_Setting{iS}_Pixel{int(info['nPix'])}.pdf")
        print(f"Saving file to {outFileName}")
        plt.savefig(outFileName, bbox_inches='tight')
        plt.close()
