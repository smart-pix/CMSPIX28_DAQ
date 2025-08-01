import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep
import argparse
import math

# plt.style.use(hep.style.ROOT)
hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath

# Gaussian function to fit
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * std_dev**2))

# def double_gaussian(x, amp1, mean1, std1, amp2, mean2, std2):
#     gauss1 = amp1 * np.exp(-((x - mean1)**2) / (2 * std1**2))
#     gauss2 = amp2 * np.exp(-((x - mean2)**2) / (2 * std2**2))
#     return gauss1 + gauss2

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
args = parser.parse_args()

# input file
inData = np.load(args.inFilePath)
features = inData["features"]

# get info
info = inspectPath(os.path.dirname(args.inFilePath))
print(info)

# output directory
outDir = args.outDir if args.outDir else os.path.dirname(args.inFilePath)
os.makedirs(outDir, exist_ok=True)
# os.chmod(outDir, mode=0o777)

pltConfig = {}
pltConfig["nelectron_asic_50perc_perBit"] = {
    "xlabel": r"S-Curve Half Max [e$^{-}$]", 
    "ylabel": r"N$_{\mathrm{Bits}}$",
    "binConfigs": [[0, 2000, 101], [800, 3000, 111], [2000, 4500, 126]], # bit 0, bit 1, bit 2
    "p0s": [[50, 400, 100], [50, 1300, 100], [50, 3300, 100]], # bit 0, bit 1, bit 2
    "fitRange" : [[250,550], [1100, 1600], [3000, 3500]],
    "ylim": [0, 35],
    "idx" : 1,
}
pltConfig["scurve_mean_perBit"] = {
    "xlabel": r"S-Curve $\mu$ [e$^{-}$]", 
    "ylabel": r"N$_{\mathrm{Bits}}$",
    "binConfigs": [[0, 2000, 101], [800, 3000, 111], [2000, 4500, 126]], # bit 0, bit 1, bit 2
    "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]], # bit 0, bit 1, bit 2
    "fitRange" : [[250,550], [1100, 1600], [3000, 3500]],
    "ylim": [0, 30],
    "idx" : 2,
}
pltConfig["scurve_std_perBit"] = {
    "xlabel": r"S-Curve $\sigma$ [e$^{-}$]", 
    "ylabel": r"N$_{\mathrm{Bits}}$",
    "binConfigs": [[0, 400, 41], [0, 400, 41], [0, 400, 41]], # bit 0, bit 1, bit 2
    "p0s": [[50, 50, 50], [50, 50, 50], [50, 50, 50]], # bit 0, bit 1, bit 2
    "fitRange" : [[20, 80], [20, 80], [20, 80]],
    "ylim": [0, 149],
    "idx" : 3,
}
print(features.shape)
for name, config in pltConfig.items():
    for iS in range(features.shape[0]):
        for iB in range(3):

            # get data to plot
            temp = features[iS:,:,iB,config["idx"]].flatten()
            if np.all(temp==-999):
                print(f"All values are -999 for feature {config['idx']}. Skipping plotting.")
                continue

            # get binning
            # bins = np.linspace(config["binConfigs"][iB][0], 
            #                    config["binConfigs"][iB][1], 
            #                    config["binConfigs"][iB][2])
            step_size = 20
            if("sigma" in config["xlabel"]):
                step_size = 10
            xlim_min, xlim_max = max(0, math.floor(min(temp) / 100) * 100 - 100), math.ceil(max(temp) / 100) * 100 + 100
            nbins = (xlim_max - xlim_min) // step_size
            bins = np.linspace(xlim_min, xlim_max, nbins + 1)  # Add 1 to include the upper limit

            # set up figure
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_xlabel(config["xlabel"], fontsize=18, labelpad=10)
            ax.set_ylabel(config["ylabel"] + f" / {bins[1]-bins[0]}" + r" e$^{-}$", fontsize=18, labelpad=10)
        
            # plot
            # print(x[iS:,:,iB:,config["idx"]])
            # print(iS, iB, config["idx"], features[iS:,:,iB:,config["idx"]].shape, features[:,iB][:,config["idx"]].shape, np.max(features[:,iB][:,config["idx"]]))
            # hist_vals, bin_edges = np.histogram(features[:,iB][:,config["idx"]], bins=bins, density=False) # inData[name][iB]
            # ax.hist(features[:,iB][:,config["idx"]], bins=bins, histtype="step", linewidth=1.5, color='black', label='Data') # plot data histogram # inData[name][iB]
            
            hist_vals, bin_edges = np.histogram(temp, bins=bins, density=False) # inData[name][iB]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.step(bin_centers, hist_vals, where='mid', linewidth=1.5, color='black', label='Data')
            # ax.hist(features[iS:,:,iB:,config["idx"]].flatten(), bins=bins, histtype="step", linewidth=1.5, color='black', label='Data') # plot data histogram # inData[name][iB]

            mean = 0
            rms = 0
            # gaussian fit
            if config["p0s"] is not None:
                
                try:
                    # set amplitude to max of hist values
                    p0 = config["p0s"][iB]
                    p0[0] = max(hist_vals)
                    mean = np.average(bin_centers, weights=hist_vals)
                    rms = np.sqrt(np.average((bin_centers - mean)**2, weights=hist_vals))
                    # restrict fit to certain region
                    mask = bin_centers != np.nan
                    if "fitRange" in config.keys():
                        print("Only fitting in the range: ", config["fitRange"][iB])
                        mask = ((bin_centers > mean - 3.0 * rms) & (bin_centers < mean + 3.0 * rms))    
                    # perform fit
                    popt, _ = curve_fit(gaussian, bin_centers[mask], hist_vals[mask], p0=p0)
                    # evaluate fit and plot
                    y_fit = gaussian(bin_centers, *popt) # evaluate gaussian at bins
                    amplitude, mean , std_dev = popt
                    std_dev = abs(std_dev) # from the gaussian the reported value could be +/- but just report positive
                    ax.plot(bin_centers[mask], y_fit[mask], color='r', label='Gaussian Fit''\n'fr'({mean:.2f},{std_dev:.2f})', alpha=0.5) # fit
                except:
                    print("Fit failed")

                # double gaussian fit
                # p01 = list(config["p0s"][iB])
                # p02 = list(config["p0s"][iB])
                # p02[0] = 5
                # p02[1] += 500 # *= 1.75
                # p0 = p01 + p02
                # print(p0)
                # popt2, _ = curve_fit(double_gaussian, bin_centers, hist_vals, p0=p0) # , bounds=bounds) # fit gaussian
                # y_fit_1 = gaussian(bin_centers, *popt2[:3]) # evaluate gaussian at bins
                # y_fit_2 = gaussian(bin_centers, *popt2[3:])
                # ax.plot(bin_centers, y_fit_1, color='b', label='Gaussian 1 Fit''\n'fr'({popt2[1]:.2f},{abs(popt2[2]):.2f})', alpha=0.5) # fit
                # ax.plot(bin_centers, y_fit_2, color='g', label='Gaussian 2 Fit''\n'fr'({popt2[4]:.2f},{abs(popt2[5]):.2f})', alpha=0.5)
                
            # add legend
            legend = ax.legend(fontsize=12)
            for text in legend.get_texts():
                text.set_fontweight('bold')

            # limits
            ax.set_xlim(bins[0], bins[-1])
            print("updated code/plots")
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
            # ax.text(0.05, 0.85, "ROIC V1, ID 11, SuperPixel 2", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
            ax.text(0.05, 0.85, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
            ax.text(0.05, 0.80, f"Bit {iB}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
            # Add text to the plot showing fitted Gaussian parameters
            # amplitude, mean , std_dev = popt
            # text = f'Bit = {iB}''\n'fr'Amplitude = {amplitude:.2f}''\n'fr'$\mu$ = {mean:.2f}''\n'fr'$\sigma$ = {std_dev:.2f}' if config["p0s"] is not None else f'Bit = {iB}'
            # ax.text(0.05, 0.8, text, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='top')
            
        
            
            
            
            # save fig
            outFileName = os.path.join(outDir, f"{name}_Setting{iS}_Bit{iB}.pdf")
            print(f"Saving file to {outFileName}")
            plt.savefig(outFileName, bbox_inches='tight')
            plt.close()
