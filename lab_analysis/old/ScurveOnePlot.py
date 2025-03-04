import numpy as np
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import glob
import matplotlib.pyplot as plt
import argparse
import matplotlib.ticker as ticker
# from scipy.integrate import cumulative_trapezoid
# from scipy.interpolate import interp1d
from pathlib import Path
import tqdm

# # Gaussian function to fit
# def gaussian(x, amplitude, mean, std_dev):
#     return amplitude * np.exp(-((x - mean)**2) / (2 * std_dev**2))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inFilePath', required=True, help='Path to input files')
    parser.add_argument("--PlotOnlyCombined", action="store_true", help="Only plot the combined s-curves")

    args = parser.parse_args()
    # list dir
    l = list(Path(args.inFilePath).glob("*"))
    Global = {}

    for nPix, i in enumerate(l):
        
        # if iL > 2: break
        if "plots" in str(i):
            continue

        nPix, vMin, vMax, vStep, nSample, vdda, VTH = map(float, re.search(r'nPix([0-9]+)_vMin([0-9.]+)_vMax([0-9.]+)_vStep([0-9.]+)_nSample([0-9.]+)_vdda([0-9.]+)_VTH([0-9.]+)', i.stem).groups())
        # vMin, vMax, vStep, nSample = map(float, re.search(r'vMin([0-9.]+)_vMax([0-9.]+)_vStep([0-9.]+)_nSample([0-9.]+)', i.stem).groups())
        pixel = int(nPix)
        files = list(i.glob("*.npz"))
        if len(files) < 81:
            print(pixel, len(files))
            continue
        Global[pixel] = files

    # convert to mV from V
    vMin *= 1000
    vMax *= 1000
    vStep *= 1000

    # outdir
    outDir = os.path.join(os.path.dirname(args.inFilePath),f"plots")
    os.makedirs(outDir, exist_ok=True)

    # Pixel programming gain - value 1-2-3
    Pgain = 1
    #input cap
    Cin = 1.85e-15
    #Electron Charge
    Qe = 1.602e-19

    # loop over
    for pixel, files in tqdm.tqdm(Global.items()):

        # one file per voltage
        v_asics = []
        data = []

        for iF, inFileName in enumerate(files):

            v_asic = float(os.path.basename(inFileName).split("vasic_")[1].split(".npz")[0])
            v_asic *= 1000 # convert v_asic to mV from V

            # pick up the data
            with np.load(inFileName) as f:
                x = f["data"]
                # only take pixel we want
                x = x.reshape(-1, 256, 3)
                x = x[:,pixel]

            # compute fraction with 1's
            frac = x.sum(0)/x.shape[0]

            # save to lists
            v_asics.append(v_asic)
            data.append(frac)
                            
        # convert
        v_asics = np.array(v_asics)
        nelectron_asics = v_asics/1000*Pgain*Cin/Qe # divide by 1000 is to convert mV to Volt
        data = np.stack(data, 1)
        
        # update global
        Global[pixel] = data
    
    # store 50% points for each bit, scurve fitted mean and std
    nelectron_asic_50perc_perBit = [[], [], []]
    scurve_mean_perBit = [[], [], []]
    scurve_std_perBit = [[], [], []]

    # filter threshold to analyse the data
    sCutHi = 0.8
    sCutLo = 0.2
    stds_threshold = 200

    # plot
    fig, ax = plt.subplots(figsize=(6,6))
    for pixel, vals in tqdm.tqdm(Global.items()):

        for iB, bit in enumerate(vals):
            
            # temp default values
            goodBit = True

            # apply threshold cuts
            if bit[0] > sCutLo or bit[-1] < sCutHi:
                print("Failed threshold cut: ", bit[0], bit[-1], sCutLo, sCutHi)
                goodBit = False
                # continue

            # perform fit
            if goodBit:
                try:
                    fitResult=curve_fit(
                        f=norm.cdf,
                        xdata=nelectron_asics,
                        ydata=bit,
                        p0=[600,30],
                        bounds=((-np.inf,0),(np.inf,np.inf))
                    )
                    mean_, std_ = fitResult[0]
                except:
                    print("fit failed")
                    mean_, std_ = -1, -1
                    goodBit = False

                # apply cuts
                if std_ > stds_threshold:
                    print("Failed std cut: ",std_, stds_threshold)
                    goodBit = False
                    # continue

            # plot and populate
            if goodBit:
                # plot
                ax.plot(nelectron_asics, bit)
                # pick up 50% values
                idx_closest = np.argmin(np.abs(bit - 0.5))
                nelectron_asic_50perc_perBit[iB].append(nelectron_asics[idx_closest])
                scurve_mean_perBit[iB].append(mean_)
                scurve_std_perBit[iB].append(std_)
            else:
                nelectron_asic_50perc_perBit[iB].append(-999)
                scurve_mean_perBit[iB].append(-999)
                scurve_std_perBit[iB].append(-999)

    # combine
    nelectron_asic_50perc_perBit = np.array(nelectron_asic_50perc_perBit)
    scurve_mean_perBit = np.array(scurve_mean_perBit)
    scurve_std_perBit = np.array(scurve_std_perBit)
    print(nelectron_asic_50perc_perBit, nelectron_asic_50perc_perBit.shape)
    # Save arrays to a file
    output_file = os.path.join(outDir, "scurve_data.npz")
    np.savez(output_file, 
             nelectron_asic_50perc_perBit=nelectron_asic_50perc_perBit, 
             scurve_mean_perBit=scurve_mean_perBit, 
             scurve_std_perBit=scurve_std_perBit)
    print(f"Data saved to {output_file}")

    # Add titles and labels
    ax.set_xlabel("Number of Electrons", fontsize=18, labelpad=10)
    ax.set_ylabel("Fraction of One's", fontsize=18, labelpad=10)
    # set limits
    ax.set_xlim(0, 3000)
    ax.set_ylim(-0.05, 1.05)
    # style ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
    ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")
    # save to file
    outFileName = os.path.join(outDir,"FullSCurve.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()

    # plot the gaussian per bit

    # initial gaussian configurations per bit
    p0s = [
        [50, 400, 100], # bit 0 (amplitude, mean, std_dev)
        [50, 1200, 100], # bit 1
        [50, 2800, 100] # bit 2
    ]
    # plot range per plots
    binConfigs = [
        [0, 800, 81], # bit 0
        [800, 1800, 101], # bit 1
        [2000, 3600, 161] # bit 2
    ]
    for iB in range(3):
        
        # set up figure
        fig, ax = plt.subplots(figsize=(6,6))
        # plot data
        bins = np.linspace(binConfigs[iB][0], binConfigs[iB][1], binConfigs[iB][2])
        ax.hist(nelectron_asic_50perc_perBit[iB], bins=bins, histtype="step")

        # plot fitten gaussian
        hist_vals, bin_edges = np.histogram(nelectron_asic_50perc_perBit[iB], bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        popt, _ = curve_fit(gaussian, bin_centers, hist_vals, p0=p0s[iB])
        y_fit = gaussian(bins, *popt) # fit gaussian
        
        ax.plot(bins, y_fit, color='r', label='Fitted Gaussian')
        # limits
        ax.set_xlim(bins[0], bins[-1])
        # ax.set_ylim(-0.05, 1.05)

        # Add titles and labels
        ax.set_xlabel(f"Bit {iB} NElectrons at 50% S-Curve", fontsize=18, labelpad=10)
        ax.set_ylabel("Number of Bits", fontsize=18, labelpad=10)
        # Set up ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
        ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")

        # Add text to the plot showing fitted Gaussian parameters
        amplitude, mean , std_dev = popt
        ax.text(0.9, 0.90, f'Amplitude = {amplitude:.2f}''\n'fr'$\mu$ = {mean:.2f}''\n'fr'$\sigma$ = {std_dev:.2f}', transform=ax.transAxes, fontsize=12, color="black", ha='right', va='center')

        # save fig
        outFileName = os.path.join(outDir,f"FullSCurve50perc_Bit{iB}.pdf")
        print(f"Saving file to {outFileName}")
        plt.savefig(outFileName, bbox_inches='tight')
        plt.close()
