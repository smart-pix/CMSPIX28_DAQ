import numpy as np
import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath, NBIT

# Analysis of the PulseDelayScan / PulseDelayScanVTH routines (CMSPIX28Spacely_Subroutines_B6_DelayScan.py)
# Input is the test folder:
#   PulseDelayScan    : <folder>/delay_<ns>.npy
#   PulseDelayScanVTH : <folder>/vth<V>/delay_<ns>.npy
# Each npy has shape (nsample, 3) with the 3 bits of the pixel for every sample
# Outputs in <folder>/plots (or -o):
#   pulse_delay_data.npz   : hit rates and timing window edges
#   PulseDelayScan_hitRate_*.pdf         : hit rate vs delay, one per threshold
#   PulseDelayScan_hitRate_vs_vth_*.pdf  : hit rate vs delay, all thresholds (VTH only)
#   PulseDelayScan_hitMap_*.pdf          : 2D hit map delay vs threshold (VTH only)
#   PulseDelayScan_window_*.pdf          : 50% timing window edges and width vs threshold (VTH only)

# Argument parser
parser = argparse.ArgumentParser(description='Analyze pulse generator delay scans.')
parser.add_argument("-i", '--inPath', type=str, required=True, help='Input test folder')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Output directory. Default is <inPath>/plots')
parser.add_argument('--contour', type=float, default=50, help='Hit rate [%%] used for the timing window edges and the 2D map contour')
args = parser.parse_args()

inPath = os.path.normpath(args.inPath)
info = inspectPath(inPath)
print(info)

# get output directory
outDir = args.outDir if args.outDir else os.path.join(inPath, "plots")
os.makedirs(outDir, exist_ok=True)

#-----------------------------------------------------------------------
# load data
#-----------------------------------------------------------------------
def loadDelayScan(path):
    files = glob.glob(os.path.join(path, "delay_*.npy"))
    files = sorted(files, key=lambda x: float(re.search(r'delay_([-\d.]+)\.npy', x).group(1)))
    delays = np.array([float(re.search(r'delay_([-\d.]+)\.npy', f).group(1)) for f in files])
    hitRate, hitRateAny, nSample = [], [], []
    for f in files:
        x = np.load(f) # (nsample, bit)
        hitRate.append(x.mean(0))
        hitRateAny.append(np.any(x, axis=1).mean())
        nSample.append(x.shape[0])
    return delays, np.array(hitRate), np.array(hitRateAny), np.array(nSample)

if info["testType"] == "PulseDelayScanVTH":
    vthDirs = [d for d in glob.glob(os.path.join(inPath, "vth*")) if os.path.isdir(d)]
    vthDirs = sorted(vthDirs, key=lambda x: float(re.search(r'vth([\d.]+)$', x).group(1)))
    vths = np.array([float(re.search(r'vth([\d.]+)$', d).group(1)) for d in vthDirs]) * 1000 # mV
else:
    vthDirs = [inPath]
    vths = np.array([-1])

delays, hitRate, hitRateAny, nSample = [], [], [], []
for d in vthDirs:
    de, hr, hra, ns = loadDelayScan(d)
    if len(de) == 0:
        print(f"No delay_*.npy files in {d}. Skipping.")
        continue
    delays.append(de)
    hitRate.append(hr)
    hitRateAny.append(hra)
    nSample.append(ns)

if len(delays) == 0:
    raise SystemExit(f"No data found in {inPath}")

# an aborted sweep can leave the last threshold with fewer delay points, keep the common ones
nDelay = min(len(d) for d in delays)
if any(len(d) != nDelay for d in delays):
    print(f"Not all thresholds have the same number of delay points, keeping the first {nDelay}")
vths = vths[:len(delays)]
delays = delays[0][:nDelay]
hitRate = 100*np.stack([h[:nDelay] for h in hitRate], 0)        # (nVth, nDelay, bit) in %
hitRateAny = 100*np.stack([h[:nDelay] for h in hitRateAny], 0)  # (nVth, nDelay) in %
nSample = np.stack([n[:nDelay] for n in nSample], 0)            # (nVth, nDelay)

# binomial uncertainty
hitRateErr = np.sqrt(hitRate*(100-hitRate)/nSample[:,:,None])
hitRateAnyErr = np.sqrt(hitRateAny*(100-hitRateAny)/nSample)

#-----------------------------------------------------------------------
# timing window: first rising and last falling crossing of args.contour
#-----------------------------------------------------------------------
def crossings(x, y, level):
    above = y >= level
    if not above.any() or above.all():
        return -999, -999
    iFirst = np.argmax(above)
    iLast = len(above) - 1 - np.argmax(above[::-1])
    # linear interpolation between neighbouring points, edge of scan if there is no neighbour
    rise = x[iFirst] if iFirst == 0 else np.interp(level, [y[iFirst-1], y[iFirst]], [x[iFirst-1], x[iFirst]])
    fall = x[iLast] if iLast == len(x)-1 else np.interp(level, [y[iLast+1], y[iLast]], [x[iLast+1], x[iLast]])
    return rise, fall

# window[iVth, iBit, (rise, fall, width)], last bit index is "any bit"
window = np.full((len(vths), NBIT+1, 3), -999.)
for iV in range(len(vths)):
    for iB in range(NBIT+1):
        y = hitRateAny[iV] if iB == NBIT else hitRate[iV,:,iB]
        rise, fall = crossings(delays, y, args.contour)
        if rise != -999:
            window[iV, iB] = [rise, fall, fall - rise]

# save
output_file = os.path.join(outDir, "pulse_delay_data.npz")
np.savez(output_file, delay_ns = delays, vth_mV = vths, hitRate = hitRate, hitRateErr = hitRateErr, hitRateAny = hitRateAny, hitRateAnyErr = hitRateAnyErr, nSample = nSample, window = window, contour = args.contour)
print(f"Data saved to {output_file}")

#-----------------------------------------------------------------------
# plots
#-----------------------------------------------------------------------
xlabel = f"Pulse Generator Delay [ns]"
baseDly = info.get("baseDly", None)
color = ["blue", "red", "orange"]

def labels(ax, extra=None, y0=0.9):
    SmartPixLabel(ax, 0.05, y0, size=22)
    ax.text(0.05, y0-0.05, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    txt = f"Pixel {int(info['nPix'])}, " if info.get("nPix") is not None else ""
    txt += f"V$_{{asic}}$ = {info['vasic']*1000:.0f} mV" if "vasic" in info else ""
    ax.text(0.05, y0-0.10, txt, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    if baseDly is not None:
        ax.text(0.05, y0-0.15, f"Delay offset from {baseDly:.1f} ns", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    if extra:
        ax.text(0.05, y0-0.20, extra, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

# hit rate vs delay, one plot per threshold
for iV, vth in enumerate(vths):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel("Hit Rate [%]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        ax.errorbar(delays, hitRate[iV,:,iB], yerr=hitRateErr[iV,:,iB], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=3)
    ax.errorbar(delays, hitRateAny[iV], yerr=hitRateAnyErr[iV], label='Any bit', color="black", linestyle='--', marker='s', markersize=3)
    ax.set_xlim(delays.min(), delays.max())
    ax.set_ylim(0, 160) # room for the labels
    legend = ax.legend(fontsize=15, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight('bold')
    SetTicks(ax)
    labels(ax, extra=f"V$_{{th}}$ = {vth:.0f} mV" if vth >= 0 else None)
    name = f"vth{vth:.0f}mV" if vth >= 0 else f"Pixel{int(info['nPix'])}"
    outFileName = os.path.join(outDir, f"PulseDelayScan_hitRate_{name}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()

# the rest needs a threshold sweep
if len(vths) < 2:
    raise SystemExit(0)

# hit rate vs delay for all thresholds
cmap = plt.cm.viridis
for iB in range(NBIT+1):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(f"Hit Rate, {'Any Bit' if iB == NBIT else f'Bit {iB}'} [%]", fontsize=18, labelpad=10)
    for iV, vth in enumerate(vths):
        y = hitRateAny[iV] if iB == NBIT else hitRate[iV,:,iB]
        ax.plot(delays, y, color=cmap(iV/(len(vths)-1)), linestyle='-', marker='o', markersize=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vths.min(), vths.max()))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"V$_{th}$ [mV]", fontsize=18)
    ax.set_xlim(delays.min(), delays.max())
    ax.set_ylim(0, 160)
    SetTicks(ax)
    labels(ax)
    outFileName = os.path.join(outDir, f"PulseDelayScan_hitRate_vs_vth_{'AnyBit' if iB == NBIT else f'Bit{iB}'}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()

# 2D hit map: x = delay, y = threshold, z = hit rate
def binEdges(c):
    # bin edges centered on the scan points
    if len(c) == 1:
        return np.array([c[0]-0.5, c[0]+0.5])
    mid = (c[1:] + c[:-1])/2
    return np.concatenate([[c[0] - (mid[0]-c[0])], mid, [c[-1] + (c[-1]-mid[-1])]])

for iB in range(NBIT+1):
    z = hitRateAny if iB == NBIT else hitRate[:,:,iB]
    fig, ax = plt.subplots(figsize=(7,6))
    cmap2D = plt.cm.viridis
    mesh = ax.pcolormesh(binEdges(delays), binEdges(vths), z, cmap=cmap2D, vmin=0, vmax=100)
    if z.min() < args.contour < z.max():
        ax.contour(delays, vths, z, levels=[args.contour], colors='white', linewidths=1.5)
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='top')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label(f"Hit Rate, {'Any Bit' if iB == NBIT else f'Bit {iB}'} [%] (white: {args.contour:.0f}%)", fontsize=18, labelpad=10)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(r"V$_{th}$ [mV]", fontsize=18, labelpad=10)
    SetTicks(ax)
    outFileName = os.path.join(outDir, f"PulseDelayScan_hitMap_{'AnyBit' if iB == NBIT else f'Bit{iB}'}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()

# timing window edges and width vs threshold
pltConfig = {
    "rise"  : {"idx" : 0, "ylabel" : f"Window Start ({args.contour:.0f}%) [ns]"},
    "fall"  : {"idx" : 1, "ylabel" : f"Window End ({args.contour:.0f}%) [ns]"},
    "width" : {"idx" : 2, "ylabel" : f"Window Width ({args.contour:.0f}%) [ns]"},
}
for name, config in pltConfig.items():
    if np.all(window[:,:,config["idx"]] == -999):
        print(f"No threshold reaches {args.contour}% hit rate. Skipping {name} plot.")
        continue
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(r"V$_{th}$ [mV]", fontsize=18, labelpad=10)
    ax.set_ylabel(config["ylabel"], fontsize=18, labelpad=10)
    ylimMax = 0
    for iB in range(NBIT+1):
        y = window[:,iB,config["idx"]]
        mask = y != -999
        if not mask.any():
            continue
        ax.plot(vths[mask], y[mask], label='Any bit' if iB == NBIT else f'Bit {iB}', color="black" if iB == NBIT else color[iB], linestyle='--' if iB == NBIT else '-', marker='o', markersize=4)
        ylimMax = max(ylimMax, np.max(y[mask]))
    ax.set_xlim(0, vths.max() * 1.1)
    ax.set_ylim(0, ylimMax * 1.6)
    legend = ax.legend(fontsize=15, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight('bold')
    SetTicks(ax)
    labels(ax)
    outFileName = os.path.join(outDir, f"PulseDelayScan_window_{name}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()
