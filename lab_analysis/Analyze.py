import numpy as np
import re
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
import glob
import argparse
import multiprocessing as mp

# Global values
Pgain = 1 # Pixel programming gain - value 1-2-3
Cin = 1.85e-15 # input cap
Qe = 1.602e-19 # electron charged
VtomV = 1000 # convert V to mV

def inspectPath(inPath):
    # pick up configurations
    split = inPath.split("/")
    info = {}
    # get the date and test type
    info["date"] = [i for i in split if "2025" in i][0].split("_")[0]
    info["time"] = [i for i in split if "2025" in i][0].split("_")[1]
    info["testType"] = [i for i in split if "2025" in i][0].split("_")[2]
    # now get all the other configurations with number values
    matches = re.findall(r'([a-zA-Z]+)([0-9.]+)', inPath)
    for match in matches:
        info[match[0]] = float(match[1])
    # convert to mV from V
    info["vMin"] *= VtomV
    info["vMax"] *= VtomV
    info["vStep"] *= VtomV
    return info

def analysis(config):

    # pick up configurations
    info = inspectPath(config["inPath"])

    # get list of files
    files = list(glob.glob(os.path.join(config["inPath"], "*.npz")))

    # one file per voltage
    v_asics = []
    data = []

    # loop over the files
    for iF, inFileName in enumerate(files):

        v_asic = float(os.path.basename(inFileName).split("vasic_")[1].split(".npz")[0])
        v_asic *= VtomV # convert v_asic to mV from V

        # pick up the data
        with np.load(inFileName) as f:
            x = f["data"]
            # only take pixel we want
            x = x.reshape(-1, 256, 3)
            x = x[:,int(info["nPix"])]

        # compute fraction with 1's
        frac = x.sum(0)/x.shape[0]

        # save to lists
        v_asics.append(v_asic)
        data.append(frac)
                        
    # convert
    v_asics = np.array(v_asics)
    nelectron_asics = v_asics/VtomV*Pgain*Cin/Qe # divide by 1000 is to convert mV to Volt
    data = np.stack(data, 1)
    
    # filter threshold to analyse the data
    sCutHi = 0.8
    sCutLo = 0.2
    stds_threshold = 200

    # loop over bits
    temp = []
    for iB, bit in enumerate(data):
        
        # temp default values
        fiftyPerc_, mean_, std_ = -999, -999, -999

        # if pass threshold, then fit and get 50% values
        if bit[0] < sCutLo and bit[-1] > sCutHi:

            # fit
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

            # pick up 50% values
            idx_closest = np.argmin(np.abs(bit - 0.5))
            fiftyPerc_ = nelectron_asics[idx_closest]

        # append
        t_ = []
        if info["testType"] == "MatrixNPix":
            t_.append(info["nPix"])
        elif info["testType"] == "MatrixVTH":
            t_.append(info["VTH"])
        else:
            t_.append(-1)
        # add other entries
        t_ += [fiftyPerc_, mean_, std_]
        # append
        temp.append(t_)

    return temp

if __name__ == "__main__":

    # user arguments
    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inFilePath', required=True, help='Path to input files')
    parser.add_argument("--PlotOnlyCombined", action="store_true", help="Only plot the combined s-curves")
    parser.add_argument('-j', '--ncpu', type=int, default=4, help='Number of CPUs to use')
    args = parser.parse_args()
    
    # outdir
    outDir = os.path.join(os.path.dirname(args.inFilePath),f"plots")
    os.makedirs(outDir, exist_ok=True)
    os.chmod(outDir, mode=0o777)

    # handle input
    inPathList = list(sorted(glob.glob(args.inFilePath)))
    inPathList = [i for i in inPathList if "plots" not in i]
    # inPathList = inPathList[:5]

    # create configurations
    confs = []
    for inPath in inPathList:
        confs.append({

            "inPath": inPath,
        })

    # launch jobs
    if args.ncpu == 1:
        results = []
        for conf in confs:
            results.append(analysis(conf))
    else:
        results = mp.Pool(args.ncpu).map(analysis, confs)

    # process
    results = np.array(results)
    print(results[:,:,0].flatten())
    output_file = os.path.join(outDir, "scurve_data.npy")
    print(f"Data saved to {output_file}")
    np.save(output_file, results)

    # plot
    # fig, ax = plt.subplots(figsize=(6,6))
    # # Add titles and labels
    # ax.set_xlabel("Number of Electrons", fontsize=18, labelpad=10)
    # ax.set_ylabel("Fraction of One's", fontsize=18, labelpad=10)
    # # set limits
    # ax.set_xlim(0, 3000)
    # ax.set_ylim(-0.05, 1.05)
    # # style ticks
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.tick_params(axis='both', which='major', labelsize=14, length=5, width=1, direction="in")
    # ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1, direction="in")
    # # save to file
    # outFileName = os.path.join(outDir,"FullSCurve.pdf")
    # print(f"Saving file to {outFileName}")
    # plt.savefig(outFileName, bbox_inches='tight')
    # plt.close()