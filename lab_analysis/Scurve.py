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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inFilePath', required=True, help='Path to input files')
    parser.add_argument("--PlotOnlyCombined", action="store_true", help="Only plot the combined s-curves")

    args = parser.parse_args()

    # pick up all of the data files
    inFileList = sorted(glob.glob(args.inFilePath))
    vMin, vMax, vStep, nSample = map(float, re.search(r'vMin([0-9.]+)_vMax([0-9.]+)_vStep([0-9.]+)_nSample([0-9.]+)', inFileList[0]).groups());
    # convert to mV from V
    vMin *= 1000
    vMax *= 1000
    vStep *= 1000

    # outdir
    outDir = os.path.join(os.path.dirname(args.inFilePath),f"plots")
    os.makedirs(outDir, exist_ok=True)

    # one file per voltage
    v_asics = []
    data = []
    for inFileName in inFileList:
        print(inFileName)
    
        v_asic = float(os.path.basename(inFileName).split("vasic_")[1].split(".npz")[0])
        v_asic *= 1000 # convert v_asic to mV from V
    
        # pick up the data
        with np.load(inFileName) as f:
            x = f["data"]
        
        # compute fraction with 1's
        frac = x.sum(0)/x.shape[0]
    
        # save to lists
        v_asics.append(v_asic)
        data.append(frac)

    v_asics = np.array(v_asics)
    data = np.stack(data, 0)
    data = data.reshape(-1, 256,3)

    # Plot each pixel and bit individually

    # plot per bit
    for iP in range(data.shape[1]):
        
        # plot per pixel
        for iB in range(data.shape[2]):

            ydata = data[:, iP, iB]
            
            # continue if no data
            if ydata.sum() == 0 or args.PlotOnlyCombined:
                continue

            print(f"Pixel {iP}, Bit {iB} has nonzero data: ", ydata)
            
            # perform fit
            try:
                fitResult=curve_fit(
                    f=norm.cdf, 
                    xdata=v_asics,
                    ydata=ydata,
                    p0=[20,1],
                    bounds=((-np.inf,0),(np.inf,np.inf))
                )
                mean_, std_ = fitResult[0]
            except:
                print("fit failed")
                mean_, std_ = -1, -1

            # make figure
            fig, ax = plt.subplots(figsize=(6,6))
                
            # Plotting the S-curve
            ax.plot(v_asics, ydata, label=f"Pixel {iP}, Bit {iB}, $\\mu$ = {mean_:.3f} $\\sigma$ = {std_:.3f}", color="black", marker="o")
            
            # Add titles and labels
            ax.set_xlabel("Voltage (mV)", fontsize=15, labelpad=10)
            ax.set_ylabel("Fraction of One's", fontsize=15, labelpad=10)
            
            # set limits
            ax.set_xlim(0, vMax*1.2)
            ax.set_ylim(-0.05, 1.05)

            # Set the number of major ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            # Add minor ticks
            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.MultipleLocator()) # 10 minor ticks per major tick
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

            # customize
            ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
            ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")

            # add legend
            # ax.legend(frameon=False, fontsize=10, loc="lower right")
            
            # add useful text
            ax.text(0.975, 0.1, f"Pixel {iP}, Bit {iB}", transform=ax.transAxes, fontsize=14, color="black", ha='right', va='center')
            ax.text(0.975, 0.05, f"$\\mu$ = {mean_:.3f}, $\\sigma$ = {std_:.3f}", transform=ax.transAxes, fontsize=14, color="black", ha='right', va='center')

            # save to file
            outFileName = os.path.join(outDir, f"s_pixel{iP}_bit{iB}.pdf")
            print(f"Saving file to {outFileName}")
            plt.savefig(outFileName, bbox_inches='tight')
            
    

    # plot all pixels together per bit
    for iB in range(data.shape[2]):
        
        # make figure
        fig, ax = plt.subplots(figsize=(6,6))

        # plot per pixel
        for iP in range(data.shape[1]):

            ydata = data[:, iP, iB]

            # continue if no data
            if ydata.sum() == 0:
                continue

            # Plotting the S-curve
            ax.plot(v_asics, ydata)

        
        # Add titles and labels
        ax.set_xlabel("Voltage (mV)", fontsize=15, labelpad=10)
        ax.set_ylabel("Fraction of One's", fontsize=15, labelpad=10)
        
        # set limits
        ax.set_xlim(0, vMax*1.2)
        ax.set_ylim(-0.05, 1.05)
        
        # Set the number of major ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        # Add minor ticks

        # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) # 10 minor ticks per major tick
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # customize
        ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
        ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")
    
        # add legend
        # ax.legend(frameon=False, fontsize=10, loc="lower right")
        
        # add useful text
        ax.text(0.975, 0.1, f"Bit {iB}", transform=ax.transAxes, fontsize=14, color="black", ha='right', va='center')
        # ax.text(0.975, 0.05, f"$\\mu$ = {mean_:.3f}, $\\sigma$ = {std_:.3f}", transform=ax.transAxes, fontsize=14, color="black", ha='right', va='center')
    
        # save to file
        outFileName = os.path.join(outDir, f"s_bit{iB}.pdf")
        print(f"Saving file to {outFileName}")
        plt.savefig(outFileName, bbox_inches='tight')
