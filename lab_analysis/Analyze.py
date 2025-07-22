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
NPIXEL = 256 # number of pixels
NBIT = 3 # number of bits

# helper function to handle in file path correctly
def handleInput(inFilePath):
    
    inPathList = inFilePath
    # handle custom path names
    if "MatrixNPix" in inPathList and "*" not in inPathList:
        inPathList = os.path.join(inPathList, "nPix*")
    elif "MatrixVTH" in inPathList and "*" not in inPathList:
        inPathList = os.path.join(inPathList, "VTH*")
    elif "MatrixInjDly" in inPathList and "*" not in inPathList:
        inPathList = os.path.join(inPathList, "injDly*")
    elif "MatrixPulseGenFall" in inPathList and "*" not in inPathList:
        inPathList = os.path.join(inPathList, "FallTime*")
    elif "MatrixCvG" in inPathList and "*" not in inPathList:
        inPathList = os.path.join(inPathList, "nPix*")
    
    # glob
    inPathList = list(sorted(glob.glob(inPathList)))
    inPathList = [i for i in inPathList if all(x not in i for x in ["plots"])]
    
    # Sort the list based on the number in the final directory
    try:
        def extract_number(path):
            match = re.search(r'(\d+(\.\d+)?)(?=\D*$)', path)
            return float(match.group()) if match else float('inf')
        inPathList.sort(key=extract_number)
    except:
        print("Could not sort the list based on the number in the final directory")
    
    # handle custom sorting
    if "MatrixPulseGenFall" in inPathList[0]:
        inPathList = sorted(inPathList, key=lambda x: float(re.search(r'FallTime([\d.eE+-]+)', x).group(1)))
    
    return inPathList

# helper function to handle output directory
def handleOutDir(outDir, inFilePath):
    
    if outDir:
        outDir = outDir
    elif "*" in inFilePath:
        outDir = os.path.join(os.path.dirname(inFilePath), f"plots")
    elif "*" not in inFilePath:
        outDir = os.path.join(inFilePath, f"plots")
    else:
        outDir = "./"
    
    os.makedirs(outDir, exist_ok=True)
    # os.chmod(outDir, mode=0o777)

    return outDir

# helper function to inspect the path and extract information
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
    # handle unique cases
    if "injDly" in split[-1]:
        info["injDly"] = int(split[-1].split("injDly")[1], 16)
    if "FallTime" in split[-1]:
        FallTime = float(split[-1].split("FallTime")[1])
        FallTime = round(FallTime, abs(int(f"{FallTime:.1e}".split("e")[1])) + 1) # handle float precision
        info["FallTime"] = FallTime
    # convert to mV from V
    if "vMin" in info.keys():
        info["vMin"] *= VtomV  
    if "vMax" in info.keys(): 
        info["vMax"] *= VtomV
    if "vStep" in info.keys():  
        info["vStep"] *= VtomV
    return info

# utility helper function to get list of files in the input path
def getFileList(inPath, extension="*.np*"):

    # make full glob path
    glob_path = os.path.join(inPath, extension)
    print(glob_path)

    # get list of files
    files = list(glob.glob(glob_path))
    files = [i for i in files if all(x not in i for x in ["steps", "settings"])] # remove settings files
    files = sorted(files) # sort

    return files

# utility helper function to load data from file list
def loadData(info, files):

    # one file per voltage
    v_asics = []
    data = []

    # loop over the files
    for iF, inFileName in enumerate(files):

        v_asic = float(os.path.basename(inFileName).split("vasic_")[1].split(".np")[0])
        v_asic *= VtomV # convert v_asic to mV from V

        # pick up the data
        if inFileName.endswith(".npy"):
            x = np.load(inFileName) # new file format
        elif inFileName.endswith(".npz"):
            with np.load(inFileName) as f:
                x = f["data"]
                # only take pixel we want
                x = x.reshape(-1, NPIXEL, NBIT)
                x = x[:,int(info["nPix"])]
        else:
            print("Not recognized file extension: ", f)

        # compute fraction with 1's
        if info["testType"] == "MatrixCalibration":
            frac = x.sum(2) / x.shape[2] # sample dimension is 2
            frac = np.transpose(frac, (1,0,2)) # sample setting, pixel, bit
        else:
            frac = x.sum(0)/x.shape[0]
        
        # save to lists
        v_asics.append(v_asic)
        data.append(frac)

    # convert
    v_asics = np.array(v_asics)
    nelectron_asics = v_asics / VtomV * Pgain * Cin / Qe # divide by 1000 is to convert mV to Volt
    data = np.stack(data, -1)

    return v_asics, nelectron_asics, data

# helper function to get features from the data
def getFeatures(data, nelectron_asics, info, doFit):
    
    # filter threshold to analyse the data
    sCutHi = 0.8
    sCutLo = 0.2
    stds_threshold = 200

    # loop over bits
    features = []
    # loop over settings
    for iS in tqdm(range(data.shape[0]), desc="Processing", unit="step"): # range(data.shape[0]):
        
        # loop over pixels
        temp_pixel = []
        for iP in range(data.shape[1]):
            
            # loop over bits
            temp_bit = []
            for iB in range(data.shape[2]):

                bit = data[iS, iP, iB]

                # temp default values
                fiftyPerc_, mean_, std_ = -999, -999, -999

                # check if good bit. if pass threshold, then fit and get 50% values
                # goodBit = True # bit[0] < sCutLo and bit[-1] > sCutHi
                # goodBit = bit[0] < sCutLo and bit[-1] > sCutHi
                goodBit = bit[-1] > sCutHi

                # fit and get 50% values
                if goodBit:
            
                    # starting p0s for bit 0, 1, 2
                    p0s = [[400, 40], [1200, 40], [2500, 40]]

                    # fit
                    if doFit:
                        try:
                            fitResult=curve_fit(
                                f=norm.cdf,
                                xdata=nelectron_asics,
                                ydata=bit,
                                p0=p0s[iB],
                                bounds=((-np.inf,0),(np.inf,np.inf))
                            )
                            mean_, std_ = fitResult[0]
                        except:
                            print("fit failed")
                            mean_, std_ = -1, -1

                    # pick up 50% values
                    idx_closest = np.argmin(np.abs(bit - 0.5))
                    fiftyPerc_ = nelectron_asics[idx_closest]

                    # pick up points closest to 0.1 and 0.9
                    idx1 = np.argmin(np.abs(bit - 0.1))
                    idx9 = np.argmin(np.abs(bit - 0.9))
                    # print(idx1, nelectron_asics[idx1], idx9, nelectron_asics[idx9])
                    fiftyPerc_ = (nelectron_asics[idx9] + nelectron_asics[idx1])/2

                # else:
                #    print("Did not pass threshold cuts: ", bit[0], bit[-1], sCutLo, sCutHi)

                # append
                t_ = []
                if info["testType"] == "MatrixNPix" or info["testType"] == "Single":
                    t_.append(info["nPix"])
                elif info["testType"] == "MatrixVTH":
                    t_.append(info["VTH"])
                elif info["testType"] == "MatrixCvG":
                    t_.append(info["nPix"])
                    t_.append(info["vth"])
                elif info["testType"] == "MatrixCalibration":
                    t_.append(-1)
                elif info["testType"] == "MatrixInjDly":
                    t_.append(info["injDly"])
                elif info["testType"] == "MatrixPulseGenFall":
                    t_.append(info["FallTime"])
                else:
                    t_.append(-1)
                # add other entries
                t_ += [fiftyPerc_, mean_, std_]
        
                # append
                temp_bit.append(t_)
            
            # append
            temp_pixel.append(temp_bit)

        # append
        features.append(temp_pixel)

    features = np.array(features)
    return features

# specialized analysis for the MatrixCvG test type
def analyze_MatrixCvG(config):
    """
    Analyze the MatrixCvG test type.
    This function assumes that the input path contains multiple vth folders.
    Each folder is analyzed separately.
    """
    v_asics, nelectron_asics, data, features = [], [], [], []
    
    # loop over each vth folder
    vthList = glob.glob(os.path.join(config["inPath"], "vth*"))
    for i in vthList:
        
        # analyze each vth folder
        info = inspectPath(i)
        files = getFileList(i, extension="*.np*")
        va, na, da = loadData(info, files)
        
        # add dimension for pixel
        da = np.expand_dims(da, axis=0)  # add pixel dimension
        da = np.expand_dims(da, axis=0)  # add setting dimension

        # get features
        fe = getFeatures(da, na, info, config["doFit"])

        # append to lists
        v_asics.append(va)
        nelectron_asics.append(na)
        data.append(da)
        features.append(fe)

    v_asics = v_asics[0]  # assume all vth checks use the same v_asic scan
    nelectron_asics = nelectron_asics[0]  # assume all vth checks use the same nelectron_asics scan
    data = np.concatenate(data, axis=0)
    features = np.concatenate(features, axis=0)

    return v_asics, nelectron_asics, data, features

# main analysis driver function
def analysis(config):

    info = inspectPath(inPath)
    print(info)

    # do custom analysis based on test type
    if "MatrixCvG" == info["testType"]:
        v_asics, nelectron_asics, data, features = analyze_MatrixCvG(config)
        return features, nelectron_asics, data
    
    # default analysis settings
    elif info["testType"] in ["Single", "MatrixNPix", "MatrixVTH", "MatrixInjDly", "MatrixPulseGenFall", "MatrixCalibration"]:
        
        info = inspectPath(inPath)
        files = getFileList(inPath)
        v_asics, nelectron_asics, data = loadData(info, files)

        # format the data
        if info["testType"] in ["Single", "MatrixNPix"]:
            data = data.reshape(1, 1, NBIT, data.shape[1])
        elif info["testType"] in ["MatrixVTH", "MatrixInjDly", "MatrixPulseGenFall"]:
            data = data.reshape(1, 1, NBIT, data.shape[1])
        else:
            print("Leaving data shape as it is for test type: ", info["testType"])

        # do analysis
        features = getFeatures(data, nelectron_asics, info, config["doFit"])

        return features, nelectron_asics, data

    return None, None, None


if __name__ == "__main__":

    # user arguments
    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inFilePath', required=True, help='Path to input files')
    parser.add_argument('-j', '--ncpu', type=int, default=1, help='Number of CPUs to use')
    parser.add_argument('-o', '--outDir', default=None, help="Output directory. If not provided then use directory of inFilePath")
    parser.add_argument('--doFit', action="store_true", help="Run the S-cruve fitting. Note the analysis will take significantly longer.")
    args = parser.parse_args()
    
    # handle output directory outdir
    outDir = handleOutDir(args.outDir, args.inFilePath)

    # handle input
    inPathList = handleInput(args.inFilePath)
    for path in inPathList:
        print(path)

    # create configurations
    confs = []
    for inPath in inPathList:
        confs.append({
            "inPath": inPath,
            "doFit" : args.doFit
        })

    # launch jobs 
    if args.ncpu == 1:
        results = []
        for conf in tqdm(confs, desc="Processing", unit="step"):
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
    features = np.concatenate(features, axis=1)
    nelectron_asics = np.array(nelectron_asics)
    scurve = np.concatenate(scurve, axis=1)

    # print out
    print("features shape (nSettings, nPixel, nBit, nFeatures): ", features.shape)
    print("nelectron_asics shape (vasic_steps): ", nelectron_asics.shape)
    print("scurve shape (nSettings, nPixel, nBit, vasic_steps): ", scurve.shape)

    # process
    output_file = os.path.join(outDir, "scurve_data.npz")
    print(f"Data saved to {output_file}")
    np.savez(
        output_file, 
        features = features, 
        nelectron_asics = nelectron_asics, 
        scurve = scurve
    )
