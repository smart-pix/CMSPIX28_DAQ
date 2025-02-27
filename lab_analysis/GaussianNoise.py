import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep

# plt.style.use(hep.style.ROOT)
hep.style.use("ATLAS")

# Gaussian function to fit
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * std_dev**2))

# pick up the data
inFile = "/mnt/local/CMSPIX28/Scurve/data/2025.02.20_SuperPixV2/plots/scurve_data.npz"
inData = np.load(inFile, allow_pickle=True)
nelectron_asic_50perc_perBit = inData["nelectron_asic_50perc_perBit"]
scurve_mean_perBit = inData["scurve_mean_perBit"]
scurve_std_perBit = inData["scurve_std_perBit"]

# output directory
outDir = os.path.dirname(inFile)

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
    "xlabel": "Number of Electrons at 50% S-Curve", 
    "ylabel": "Number of Bits",
    "binConfigs": [[0, 800, 81], [800, 1800, 101], [2000, 3600, 161]], # bit 0, bit 1, bit 2
    "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]] # bit 0, bit 1, bit 2
}
pltConfig["scurve_mean_perBit"] = {
    "xlabel": "Mean of S-Curve", 
    "ylabel": "Number of Bits",
    "binConfigs": [[0, 800, 81], [800, 1800, 101], [2000, 3600, 161]], # bit 0, bit 1, bit 2
    "p0s": [[50, 400, 100], [50, 1200, 100], [50, 2800, 100]] # bit 0, bit 1, bit 2
}
pltConfig["scurve_std_perBit"] = {
    "xlabel": "Standard Deviation of S-Curve", 
    "ylabel": "Number of Bits",
    "binConfigs": [[0, 300, 31], [0, 300, 31], [0, 300, 31]], # bit 0, bit 1, bit 2
    "p0s": None # bit 0, bit 1, bit 2
}
# ["nelectron_asic_50perc_perBit", "scurve_mean_perBit", "scurve_std_perBit"]:
for name, config in pltConfig.items():
    for iB in range(3):
        
        # get binning
        bins = np.linspace(config["binConfigs"][iB][0], 
                           config["binConfigs"][iB][1], 
                           config["binConfigs"][iB][2])

        # set up figure
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlabel(config["xlabel"], fontsize=18, labelpad=10)
        ax.set_ylabel(config["ylabel"], fontsize=18, labelpad=10)
        
        # plot
        ax.hist(inData[name][iB], bins=bins, histtype="step") # plot data histogram
        
        # gaussian fit
        if config["p0s"] is not None:
            hist_vals, bin_edges = np.histogram(inData[name][iB], bins=bins, density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            popt, _ = curve_fit(gaussian, bin_centers, hist_vals, p0=config["p0s"][iB]) # fit gaussian
            y_fit = gaussian(bin_centers, *popt) # evaluate gaussian at bins
            ax.plot(bin_centers, y_fit, color='r', label='Fitted Gaussian') # fit
        
        # limits
        ax.set_xlim(bins[0], bins[-1])
        # ax.set_ylim(-0.05, 1.05)
        # Set up ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
        ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")

        # Add text to the plot showing fitted Gaussian parameters
        amplitude, mean , std_dev = popt
        text = f'Bit = {iB}''\n'fr'Amplitude = {amplitude:.2f}''\n'fr'$\mu$ = {mean:.2f}''\n'fr'$\sigma$ = {std_dev:.2f}' if config["p0s"] is not None else f'Bit = {iB}'
        ax.text(0.9, 0.90, text, transform=ax.transAxes, fontsize=12, color="black", ha='right', va='center')

        # save fig
        outFileName = os.path.join(outDir, f"{name}_Bit{iB}.pdf")
        print(f"Saving file to {outFileName}")
        plt.savefig(outFileName, bbox_inches='tight')
        plt.close()