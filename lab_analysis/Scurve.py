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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


# Gaussian function to fit
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * std_dev**2))

def scurves_to_gauss(y_values, x_values, bit, p0, outFileName):
    # Store 50% points for each curve
    fifty_percent_points = []

    for y in y_values:

        # Find the index where y_value is closest to 0.5
        idx_closest = np.argmin(np.abs(y - 0.5))
            
        # Get the corresponding x-value (50% point)
        fifty_percent_point = x_values[idx_closest]
        fifty_percent_points.append(fifty_percent_point)
    
    # Convert list to numpy array for easier manipulation
    fifty_percent_points = np.array(fifty_percent_points)
    
    # Plot histogram of the 50% points (should resemble a Gaussian)
    plt.hist(fifty_percent_points, bins=20, density=True, alpha=0.6, color='b', label='50% points')

    # Histogram data for fitting
    hist_vals, bin_edges = np.histogram(fifty_percent_points, bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit the histogram to a Gaussian
    popt, _ = curve_fit(gaussian, bin_centers, hist_vals, p0=p0)

    # Plot the fitted Gaussian
    x_fit = np.linspace(min(fifty_percent_points), max(fifty_percent_points), 1000)
    y_fit = gaussian(x_fit, *popt)
    plt.plot(x_fit, y_fit, color='r', label='Fitted Gaussian')

    # Add text to the plot showing fitted Gaussian parameters
    amplitude, mean , std_dev = popt
    textstr = f'Fit parameters:\nMean = {mean:.2f}\nAmplitude = {amplitude:.2f}\nStd Dev = {std_dev:.2f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Add labels and legend
    plt.xlabel('50% points of S-curves')
    plt.ylabel('Density')
    plt.legend()
    # plt.title('Histogram of 50% points with Gaussian Fit')
    print(outFileName)
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()
    #plt.show()

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

    # filter threshold to analyse the data 
    # ydataCutHi =0.8
    # ydataCutLo =0.2
    # stds_threshold =10
    ydataCutHi =0.3
    ydataCutLo =0.5
    stds_threshold =10000
    # outdir
    outDir = os.path.join(os.path.dirname(args.inFilePath),f"plots")
    os.makedirs(outDir, exist_ok=True)

    # Pixel programming gain - value 1-2-3
    Pgain = 1
    #input cap
    Cin = 1.85e-15
    #Electron Charge
    Qe = 1.602e-19
     
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
    nelectron_asics = v_asics/1000*Pgain*Cin/Qe #divide by 1000 is to convert mV to Volt
    data = np.stack(data, 0)
    data = data.reshape(-1, 256,3)

    # Plot each pixel and bit individually
    means = []
    stds = []

    # plot per bit
    for iP in range(data.shape[1]):
        
        means.append([-1] * data.shape[2])
        stds.append([-1] * data.shape[2])

        # plot per pixel
        for iB in range(data.shape[2]):

            ydata = data[:, iP, iB]
            
            # continue if no data
            if ydata.sum() == 0 or args.PlotOnlyCombined or ydata[-1]<ydataCutHi or ydata[0]>ydataCutLo:
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
            
            # append mean and std
            means[-1][iB] = mean_
            stds[-1][iB] = std_
            if std_ > stds_threshold:
                continue
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
            

    # save mean and std
    means = np.array(means)
    stds = np.array(stds)
    outMeanStdName = os.path.join(outDir, f"s_fit_values.npz")
    print(f"Saving fitted mean and std values to {outMeanStdName} with shapes mean {means.shape}, std {stds.shape}")
    np.savez(outMeanStdName, **{"mean": means, "std" : stds})

    # plot all pixels together per bit
    for iB in range(data.shape[2]):
        
        # make figure
        fig, ax = plt.subplots(figsize=(6,6))
        temp = []
        fiftyChargeValue = []
        # plot per pixel
        for iP in range(data.shape[1]):

            ydata = data[:, iP, iB]
            
            # Find the index where y_value is closest to 0.5
            #idx_closest = np.argmin(np.abs(ydata - 0.5))
            # Get the corresponding x-value (50% point)
            #fiftyChargeValue.append(nelectron_asics[idx_closest])
            fiftyChargeValue.append(stds[iP,iB])

            # continue if no data
            if ydata.sum() == 0 or ydata[-1]<ydataCutHi or ydata[0]>ydataCutLo:
                continue

            # Plotting the S-curve
            if stds[iP][iB] > stds_threshold:
                continue
            
            ax.plot(v_asics, ydata)
            temp.append(ydata)

       
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

        plt.close()
        #Gaussian Fit PLot
        startingMu = [1000, 2500,  3500]
        print(len(temp))
        scurves_to_gauss(temp,nelectron_asics, iB, [1,startingMu[iB], 100],os.path.join(outDir,f"Qdispersion{iB}.pdf") )

        # plotting 50% value for each pixel in their order
        # make figure
        fig, ax = plt.subplots(figsize=(6,6))
            
        # Plotting the S-curve
        fiftyChargeValue = np.array(fiftyChargeValue)
        print(fiftyChargeValue)
        print(fiftyChargeValue.shape)
        ax.plot(np.arange(fiftyChargeValue.shape[0]), fiftyChargeValue, marker="o")

        # Add titles and labels
        ax.set_xlabel("Pixel", fontsize=15, labelpad=10)
        ax.set_ylabel("Charge", fontsize=15, labelpad=10)

        # set limits
        ax.set_xlim(0, 256)
        # ax.set_ylim(-0.05, 1.05)

        # Set the number of major ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # Add minor ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) # 10 minor ticks per major tick
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # customize
        ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
        ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")

        # add legend
        # ax.legend(frameon=False, fontsize=10, loc="lower right")

        # add useful text
        ax.text(0.975, 0.1, f"Bit {iB}", transform=ax.transAxes, fontsize=14, color="black", ha='right', va='center')

        # save to file
        outFileName = os.path.join(outDir, f"SCurve_StdVsPixel_bit{iB}.pdf")
        print(f"Saving file to {outFileName}")
        plt.savefig(outFileName, bbox_inches='tight')

l





