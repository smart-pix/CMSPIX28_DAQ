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

# Gaussian function to fit
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * std_dev**2))

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFile', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default="./plots", help='Input file path')
args = parser.parse_args()

# pick up the data
# inFile = "/mnt/local/CMSPIX28/Scurve/data/2025.02.20_SuperPixV2/plots/scurve_data.npz"
# inData = np.load(args.inFile)
# nelectron_asic_50perc_perBit = inData[:,:,1] # inData["nelectron_asic_50perc_perBit"]
# scurve_mean_perBit = inData[:,:,2] # inData["scurve_mean_perBit"]
# scurve_std_perBit = inData[:,:,3] # inData["scurve_std_perBit"]

# input file
x = np.load(args.inFile)
print(x.shape)
info = inspectPath(os.path.dirname(args.inFile))

# output directory
outDir = "./plots" # os.path.dirname(inFile)

# we want to plot matrix bit order vs mean
# fig, ax = plt.subplots(figsize=(6,6))
# temp_x = np.linspace(0, 767, 768)
# temp_y = scurve_std_perBit.T.flatten()
# # print(temp_y)
# idx = temp_y>0
# ax.scatter(temp_x[idx], temp_y[idx])
# # ax.set_ylim(0,300)
# plt.savefig(os.path.join(outDir, "test.pdf"), bbox_inches='tight')

pltConfig = {}
pltConfig["nelectron_asic_50perc_perBit"] = {
    "xlabel": r"S-Curve Half Max [e$^{-}$]", 
    "ylabel": r"N$_{\mathrm{Bits}}$",
    "binConfigs": [[0, 800, 41], [800, 1600, 41], [2000, 3600, 81]], # bit 0, bit 1, bit 2
    "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]], # bit 0, bit 1, bit 2
    "ylim": [0, 30],
    "idx" : 1,
}
pltConfig["scurve_mean_perBit"] = {
    "xlabel": r"S-Curve $\mu$ [e$^{-}$]", 
    "ylabel": r"N$_{\mathrm{Bits}}$",
    "binConfigs": [[0, 800, 41], [800, 1600, 41], [2000, 3600, 81]], # bit 0, bit 1, bit 2
    "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]], # bit 0, bit 1, bit 2
    "ylim": [0, 30],
    "idx" : 2,
}
pltConfig["scurve_std_perBit"] = {
    "xlabel": r"S-Curve $\sigma$ [e$^{-}$]", 
    "ylabel": r"N$_{\mathrm{Bits}}$",
    "binConfigs": [[0, 300, 31], [0, 300, 31], [0, 300, 31]], # bit 0, bit 1, bit 2
    "p0s": None, # bit 0, bit 1, bit 2
    "ylim": [0, 100],
    "idx" : 3,
}

for name, config in pltConfig.items():
    for iB in range(3):
        
        # get binning
        bins = np.linspace(config["binConfigs"][iB][0], 
                           config["binConfigs"][iB][1], 
                           config["binConfigs"][iB][2])

        # set up figure
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlabel(config["xlabel"], fontsize=18, labelpad=10)
        ax.set_ylabel(config["ylabel"] + f" / {bins[1]-bins[0]}" + r" e$^{-}$", fontsize=18, labelpad=10)
        
        # plot
        print(iB, config["idx"], x[:,iB:,config["idx"]].shape, x[:,iB][:,config["idx"]].shape)
        hist_vals, bin_edges = np.histogram(x[:,iB][:,config["idx"]], bins=bins, density=False) # inData[name][iB]
        ax.hist(x[:,iB][:,config["idx"]], bins=bins, histtype="step", linewidth=1.5, color='black', label='Data') # plot data histogram # inData[name][iB]
        
        # gaussian fit
        if config["p0s"] is not None:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            popt, _ = curve_fit(gaussian, bin_centers, hist_vals, p0=config["p0s"][iB]) # fit gaussian
            amplitude, mean , std_dev = popt
            y_fit = gaussian(bin_centers, *popt) # evaluate gaussian at bins
            ax.plot(bin_centers, y_fit, color='r', label='Gaussian Fit''\n'fr'({mean:.2f},{std_dev:.2f})') # fit
        
        # add legend
        legend = ax.legend(fontsize=12)
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # limits
        ax.set_xlim(bins[0], bins[-1])
        if config["ylim"] is not None:
            ax.set_ylim(config["ylim"])
        else:
            ax.set_ylim(0, 1.25 * max(hist_vals))


        # ax.set_ylim(-0.05, 1.05)
        # Set up ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=1, direction="in", pad=8)
        ax.tick_params(axis='both', which='minor', labelsize=18, length=4, width=1, direction="in", pad=8)

        # add label and text
        SmartPixLabel(ax, 0.05, 0.9, size=22)
        ax.text(0.05, 0.85, "ROIC V1, ID 11, SuperPixel 2", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
        # Add text to the plot showing fitted Gaussian parameters
        amplitude, mean , std_dev = popt
        # text = f'Bit = {iB}''\n'fr'Amplitude = {amplitude:.2f}''\n'fr'$\mu$ = {mean:.2f}''\n'fr'$\sigma$ = {std_dev:.2f}' if config["p0s"] is not None else f'Bit = {iB}'
        # ax.text(0.9, 0.90, text, transform=ax.transAxes, fontsize=12, color="black", ha='right', va='center')

        

       

        # save fig
        outFileName = os.path.join(outDir, f"{name}_Bit{iB}.pdf")
        print(f"Saving file to {outFileName}")
        plt.savefig(outFileName, bbox_inches='tight')
        plt.close()
