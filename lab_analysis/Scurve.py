import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import glob
import matplotlib.pyplot as plt

# pick up all of the data files
inFilePath = "/asic/projects/C/CMS_PIX_28/benjamin/testing/workarea/CMSPIX28_DAQ/spacely/PySpacely/data_*.npz"
inFileList = sorted(glob.glob(inFilePath))

# one file per voltage
v_asics = []
data = []
for inFileName in inFileList:
    print(inFileName)
    
    v_asic = float(os.path.basename(inFileName).split("data_")[1].split(".npz")[0])

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

# plot and fit
mean = []
std = []
cov = []
for iP in range(data.shape[1]):
    
    m = []
    s = []
    c = []

    for iB in range(data.shape[2]):

        ydata = data[:, iP, iB]
        
        if ydata.sum() != 0:
            print(f"Pixel {iP}, Bit {iB} has nonzero data: ", ydata)

        #fit the CDF of the data
        fitResult=curve_fit(
            f=norm.cdf, 
            xdata=v_asics,
            ydata=ydata,
            p0=[1,1],
            bounds=((-np.inf,0),(np.inf,np.inf))
        )
        mean_, std_=fitResult[0]

        m.append(mean_)
        s.append(std_)
        c.append(fitResult[1])

        # make figure
        fig, ax = plt.subplots(figsize=(6,6))

        # Plotting the S-curve
        ax.plot(v_asics, ydata, label=f"Pixel {iP}, Bit {iB}, $\\mu$ = {mean_:.3f} $\\sigma$ = {std_:.3f}")

        # Add titles and labels
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Fraction of One's")

        # Save the plot to a file
        plt.legend()
        plt.savefig(f"plots/s_pixel{iP}_bit{iB}.pdf", bbox_inches='tight')

    mean.append(m)
    std.append(s)
    cov.append(c)

# convert to np
mean = np.array(mean)
std = np.array(std)
cov = np.array(cov)
print(mean.shape, std.shape, cov.shape)