import numpy as np
import re
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
import glob
import argparse
import multiprocessing as mp
from tqdm import tqdm 

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
    print(info)

    # get list of files
    files = list(glob.glob(os.path.join(config["inPath"], "*.np*")))

    # one file per voltage
    v_asics = []
    data = []

    # loop over the files
    for iF, inFileName in tqdm(enumerate(files), desc="Processing", unit="step"):

        v_asic = float(os.path.basename(inFileName).split("vasic_")[1].split(".np")[0])
        v_asic *= VtomV # convert v_asic to mV from V

        # pick up the data
        if inFileName.endswith(".npy"):
            x = np.load(inFileName) # new file format
        elif inFileName.endswith(".npz"):
            with np.load(inFileName) as f:
                x = f["data"]
                # only take pixel we want
                x = x.reshape(-1, 256, 3)
                x = x[:,int(info["nPix"])]
        else:
            print("Not recognized file extension: ", f)

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
        if info["testType"] == "MatrixNPix" or info["testType"] == "Single":
            t_.append(info["nPix"])
        elif info["testType"] == "MatrixVTH":
            t_.append(info["VTH"])
        else:
            t_.append(-1)
        # add other entries
        t_ += [fiftyPerc_, mean_, std_]
        # append
        temp.append(t_)
    # print(data.shape, nelectron_asics.shape, np.array(temp).shape)
    return temp, nelectron_asics, data

if __name__ == "__main__":

    # user arguments
    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inFilePath', required=True, help='Path to input files')
    parser.add_argument('-j', '--ncpu', type=int, default=1, help='Number of CPUs to use')
    parser.add_argument('-o', '--outDir', default=None, help="Output directory. If not provided then use directory of inFilePath")
    args = parser.parse_args()
    
    # outdir
    outDir = args.outDir if args.outDir else os.path.join(os.path.dirname(args.inFilePath), f"plots")
    os.makedirs(outDir, exist_ok=True)
    os.chmod(outDir, mode=0o777)

    # handle input
    inPathList = list(sorted(glob.glob(args.inFilePath)))
    inPathList = [i for i in inPathList if "plots" not in i]
    
    # Sort the list based on the number in the final directory
    try:
        def extract_number(path):
            match = re.search(r'(\d+(\.\d+)?)(?=\D*$)', path)
            return float(match.group()) if match else float('inf')
        inPathList.sort(key=extract_number)
    except:
        print("Could not sort the list based on the number in the final directory")

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

    # split up
    features = []
    nelectron_asics = results[0][1] # same for all inputs
    scurve = []
    for i,j,k in results:
        features.append(i)
        scurve.append(k)
    features = np.array(features)
    nelectron_asics = np.array(nelectron_asics)
    scurve = np.array(scurve)
    print(features.shape, nelectron_asics.shape, scurve.shape)

    # process
    # results = np.array(results)
    # output_file = os.path.join(outDir, "scurve_data.npy")
    # print(f"Data saved to {output_file}")
    # np.save(output_file, results)

    # process
    output_file = os.path.join(outDir, "scurve_data.npz")
    print(f"Data saved to {output_file}")
    np.savez(
        output_file, 
        features = features, 
        nelectron_asics = nelectron_asics, 
        scurve = scurve
    )
