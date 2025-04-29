# python imports
import re
import numpy as np 
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mplhep as hep
hep.style.use("ATLAS")
import argparse
import glob
from PyPDF2 import PdfMerger

# custom code
from SmartPixStyle import *
from Analyze import inspectPath

# takes in the features and path with individual pixel data
def analyze(features, inPixPath):
    
    # get information
    info = inspectPath(inPixPath)
    print(info)

    # load the setting file
    settingsPath = os.path.join(inPixPath, "settings.npy")
    settings = np.load(settingsPath)

    # mu_c is the pixel count per settings that respect the condition mu_bit2 > mu_bit1 > mu_bit0 > 0
    mu_c = np.zeros((features.shape[0]))
    fifty_c = np.zeros((features.shape[0]))
    
    # loop over the settings
    iP = 0 # only one pixel
    fIdx = 1 # index of 50% values in features
    muIdx = 2 # index of mu in features
    for i in range(features.shape[0]):
        mu_c[i] = int((features[i,iP,2,muIdx] > 1.5*features[i,iP,1,muIdx]) & (features[i,iP,1,muIdx] > 1.5*features[i,iP,0,muIdx]) & (features[i,iP,0,muIdx] > 0))
        fifty_c[i] = int((features[i,iP,2,fIdx] > 1.5*features[i,iP,1,fIdx]) & (features[i,iP,1,fIdx] > 1.5*features[i,iP,0,fIdx]) & (features[i,iP,0,fIdx] > 0))
        
    bestSettingResult = np.max(mu_c)
    best_settings = np.argmax(mu_c)
    top_10_indices = np.argsort(mu_c)[-10:][::-1]
    nWorkingSetting = np.count_nonzero(mu_c>0)
    
    # print(nWorkingSetting)
    # print(f"best setting index = {best_settings}, number of working pixel = {bestSettingResult}, setting = {settings[best_settings]}" )
    # print(f"top 10 settings = {top_10_indices}" )
    # print(f"top 10 settings = {mu_c[top_10_indices]}" )
    # print(f"mu bit2 for 5 pixels  = {features[best_settings,:,2,2][0:5]}" )
    # print(f"mu bit1 for 5 pixels  = {features[best_settings,:,1,2][0:5]}" )
    # print(f"mu bit0 for 5 pixels  = {features[best_settings,:,0,2][0:5]}" )
    # print(f"sigma bit2 for 5 pixels  = {features[best_settings,:,2,3][0:5]}" )
    # print(f"sigma bit2 for 5 pixels  = {features[best_settings,:,1,3][0:5]}" )
    # print(f"sigma bit2 for 5 pixels  = {features[best_settings,:,0,3][0:5]}" )
    
    mu_c = np.array(mu_c)
    testSample = settings[:,4]
    testDelay = settings[:,5]
    
    # convert to int
    testSample = np.vectorize(lambda x: int(x, 16))(testSample)
    testDelay = np.vectorize(lambda x: int(x, 16))(testDelay)

    # plot
    fig, ax = plt.subplots()
    sc = ax.scatter(testDelay, testSample, c=mu_c, cmap='viridis', edgecolor='black', marker="s") # s=40
    ax.set_xlabel('cfg_test_delay', fontsize=18, labelpad=10)
    ax.set_ylabel('cfg_test_sample', fontsize=18, labelpad=10)
    ax.set_xlim(min(testDelay)-1, max(testDelay)+1)
    ax.set_ylim(min(testSample)-1, max(testSample)+1)
    ax.set_aspect('auto')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Working Setting', fontsize=18)
    cb.set_ticks([0, 1])
    fig.tight_layout()
    
    # set ticks
    SetTicks(ax, nbinsMajor=10)

    # add label and text
    SmartPixLabel(ax, 0, 1.005, size=15)
    ax.text(1, 1.005, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}, Pixel {int(info['nPix'])}, Working Settings {nWorkingSetting}", transform=ax.transAxes, fontsize=10, color="black", ha='right', va='bottom')

    # get output directory
    outDir = os.path.join(inPixPath, "plots")
    os.makedirs(outDir, exist_ok=True)

    # save fig
    outFileName = os.path.join(outDir, f"SCurve_ChipVersion{int(info['ChipVersion'])}_ChipID{int(info['ChipID'])}_SuperPixel{int(info['SuperPix'])}_nPix{int(info['nPix'])}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')

    return outFileName

def concatenate_pdfs(pdf_paths, output_path):
    merger = PdfMerger()
    for pdf in pdf_paths:
        merger.append(pdf)
    merger.write(output_path)
    merger.close()

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
    parser.add_argument("-o", '--outDir', type=str, default=None, help='Input file path')
    parser.add_argument("--combine", action="store_true", help="Combine all pdfs into single file")
    args = parser.parse_args()

    # get file information
    info = inspectPath(os.path.dirname(args.inFilePath))
    print(info)
    
    # scurve data
    inData = np.load(os.path.join(args.inFilePath, "plots/scurve_data.npz"))
    features = inData["features"]
    nelectron_asics = inData["nelectron_asics"]
    scurve = inData["scurve"]
    print("scurve shape = ", scurve.shape)
    
    # handle input
    nPixPath = os.path.join(args.inFilePath, "nPix*")
    inPixList = list(sorted(glob.glob(nPixPath), key=lambda p: int(re.search(r'nPix(\d+)', p).group(1))))
    inPixList = [i for i in inPixList if all(x not in i for x in ["plots"])]
    print("Number of nPix paths: ", len(inPixList))
    
    # loop over pixels
    nPixels = scurve.shape[1]
    outPdfPaths = []
    for iP, inPixPath in enumerate(inPixList):
        pixNum = int(os.path.basename(inPixPath).strip("nPix"))
        temp = np.expand_dims(features[:,pixNum], axis=1)
        outPdfPath = analyze(temp, inPixPath)
        outPdfPaths.append(outPdfPath)

    # append all pdfs into one
    if args.combine:
        print(f"Merging {len(outPdfPaths)} PDFs")
        combinedPdfPath = os.path.join(args.inFilePath, "plots", f"SCurve_ChipVersion{int(info['ChipVersion'])}_ChipID{int(info['ChipID'])}_SuperPixel{int(info['SuperPix'])}_CombinedPixels.pdf")
        concatenate_pdfs(outPdfPaths, combinedPdfPath)
