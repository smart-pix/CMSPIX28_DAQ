import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep

# plt.style.use(hep.style.ROOT)
hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath

inFile = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2/2025.02.27_17.45.00_MatrixVTH_vMin0.001_vMax0.400_vStep0.00100_nSample1000.000_vdda0.900_BXCLK10.00_nPix0/plots/scurve_data.npy"
x = np.load(inFile)
info = inspectPath(os.path.dirname(inFile))
print(info)
outDir = "./plots" # os.path.dirname(inFile)

# plot config
pltConfig = {}
pltConfig["nelectron_asic_50perc"] = {
    "xlabel": r"V$_{\mathrm{TH}}$ [mV]", 
    "ylabel": r"S-Curve Half Max [e$^{-}$]", 
    "idx" : 1,
    "fit" : "linear"
    # "binConfigs": [[0, 800, 41], [800, 1600, 41], [2000, 3400, 71]], # bit 0, bit 1, bit 2
    # "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]], # bit 0, bit 1, bit 2
    # "ylim": [0, 30]
}
pltConfig["scurve_mean"] = {
    "xlabel": r"V$_{\mathrm{TH}}$ [mV]", 
    "ylabel": r"S-Curve $\mu$ [e$^{-}$]",
    "idx" : 2,
    "fit" : "linear"
    # "binConfigs": [[0, 800, 41], [800, 1600, 41], [2000, 3400, 71]], # bit 0, bit 1, bit 2
    # "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]], # bit 0, bit 1, bit 2
    # "ylim": [0, 30]
}
pltConfig["scurve_std"] = {
    "xlabel": r"V$_{\mathrm{TH}}$ [mV]",
    "ylabel": r"S-Curve $\sigma$ [e$^{-}$]",
    "idx" : 3,
    "fit" : None
    # "binConfigs": [[0, 300, 31], [0, 300, 31], [0, 300, 31]], # bit 0, bit 1, bit 2
    # "p0s": None, # bit 0, bit 1, bit 2
    # "ylim": [0, 60]
}

# Perform linear fit
def linear_func(x, a, b):
    return a * x + b

# make plots
for name, config in pltConfig.items():

    # set up figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(config["xlabel"], fontsize=18, labelpad=10)
    ax.set_ylabel(config["ylabel"], fontsize=18, labelpad=10)

    # set y limit
    maxValue = np.max(x[:,:,config["idx"]])
    ylimMax = 1.25 * maxValue
    ax.set_ylim(0, ylimMax)

    # plot
    color = ["blue", "red", "orange"]
    for iB in range(3):
        x_ = x[:,iB][:,0]
        y_ = x[:,iB][:,config["idx"]]
        mask = y_ > 0
        ax.plot(x_[mask], y_[mask], label=f'Bit {iB}', color=color[iB], marker='o', linestyle='-', markersize=4)
        
        # linear fit
        if config["fit"] == "linear":
            popt, pcov = curve_fit(linear_func, x_[mask], y_[mask])
            a, b = popt
            fit_label = f'Fit: y = {a:.2f}x + {b:.2f}'
            ax.plot(x_[mask], linear_func(x_[mask], *popt), linestyle='--', color = color[iB], alpha=0.5)
            # Calculate the angle of the line for rotation
            # Get axis scaling factors
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Compute the scale-adjusted slope
            aspect_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            adjusted_slope = a / aspect_ratio
            # Compute the angle in degrees
            angle = np.arctan(adjusted_slope) * (180 / np.pi)
            # plot
            x_middle = x_[mask][0] + (x_[mask][-1] - x_[mask][0]) / 4
            y_middle = linear_func(x_middle, *popt)
            y_middle += 0.1*y_middle
            ax.text(x_middle, y_middle, fit_label, fontsize=12, color=color[iB], ha='left', va='bottom', rotation=angle, alpha=0.5)

    # make legend
    legend = ax.legend(fontsize=15, loc="upper right")
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