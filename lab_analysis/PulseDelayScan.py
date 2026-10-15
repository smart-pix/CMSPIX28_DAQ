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

# Analysis of the pulse generator delay scans (CMSPIX28Spacely_Subroutines_B6_DelayScan.py)
# Input is the test folder:
#   PulseDelayScan       : <folder>/delay_<ns>.npy
#   PulseDelayScanVTH    : <folder>/vth<V>/delay_<ns>.npy
#   PulseDelayScanMatrix : <folder>/nPix<n>/vth<V>/delay_<ns>.npy
# Each npy has shape (nsample, 3) with the 3 bits of the pixel for every sample
# Outputs in <folder>/plots (or -o):
#   PulseDelayScan       : PulseDelayScan_hitRate_Pixel<n>.pdf (hit rate vs delay per bit), pulse_delay_data.npz
#   PulseDelayScanVTH    : PulseDelayScan_hitMap_Bit<b>.pdf (2D hit map delay vs threshold per bit)
#                          PulseDelayScan_cumulativeHitRate.pdf (hit rate over all delays vs threshold per bit)
#                          pulse_delay_data.npz
#   PulseDelayScanMatrix : the same two plots per pixel in nPix<n>/ (skip with --noPixelPlots)
#                          PulseDelayScan_widthAtHalfMaxVth_map_Bit<b>.pdf (pixel map) and _hist.pdf
#                          pulse_delay_matrix_data.npz
# Width at half maximum threshold (matrix only), per pixel and bit:
#   maximum threshold Vmax = highest threshold at which the hit rate reaches 50% at some delay
#   width = delay range [ns] with hit rate >= 50% at Vmax/2 (interpolated between the scanned thresholds)

# Argument parser
parser = argparse.ArgumentParser(description='Analyze pulse generator delay scans.')
parser.add_argument("-i", '--inPath', type=str, required=True, help='Input test folder')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Output directory. Default is <inPath>/plots')
parser.add_argument('--contour', type=float, default=50, help='Hit rate [%%] used for the 2D map contour and the width at half maximum threshold')
parser.add_argument('--noPixelPlots', action="store_true", help='PulseDelayScanMatrix: only make the matrix summary plots, not the plots for every pixel')
args = parser.parse_args()

inPath = os.path.normpath(args.inPath)
info = inspectPath(inPath)
print(info)

# get output directory
outDir = args.outDir if args.outDir else os.path.join(inPath, "plots")
os.makedirs(outDir, exist_ok=True)

color = ["blue", "red", "orange"]
xlabel = "Pulse Generator Delay [ns]"
baseDly = info.get("baseDly", None)

# pixel layout used by MatrixNPix2D.py: grid[row][column] = pixel number, row 0 at the top
grid = [
    [63,  62,  59,  58,  55,  54,  51,  50,  47,  46,  43,  42,  39,  38,  35,  34,  31,  30,  27,  26,  23,  22,  19,  18,  15,  14,  11,  10,   7,   6,   3,    2],
    [60,  61,  56,  57,  52,  53,  48,  49,  44,  45,  40,  41,  36,  37,  32,  33,  28,  29,  24,  25,  20,  21,  16,  17,  12,  13,   8,   9,   4,   5,   0,    1],
    [127, 126, 123, 122, 119, 118, 115, 114, 111, 110, 107, 106, 103, 102, 99,  98,  95,  94,  91,  90,  87,  86,  83,  82,  79,  78,  75,  74,  71,  70,  67,   66],
    [124, 125, 120, 121, 116, 117, 112, 113, 108, 109, 104, 105, 100, 101, 96,  97,  92,  93,  88,  89,  84,  85,  80,  81,  76,  77,  72,  73,  68,  69,  64,   65],
    [191, 190, 187, 186, 183, 182, 179, 178, 175, 174, 171, 170, 167, 166, 163, 162, 159, 158, 155, 154, 151, 150, 147, 146, 143, 142, 139, 138, 135, 134, 131, 130],
    [188, 189, 184, 185, 180, 181, 176, 177, 172, 173, 168, 169, 164, 165, 160, 161, 156, 157, 152, 153, 148, 149, 144, 145, 140, 141, 136, 137, 132, 133, 128, 129],
    [255, 254, 251, 250, 247, 246, 243, 242, 239, 238, 235, 234, 231, 230, 227, 226, 223, 222, 219, 218, 215, 214, 211, 210, 207, 206, 203, 202, 199, 198, 195, 194],
    [252, 253, 248, 249, 244, 245, 240, 241, 236, 237, 232, 233, 228, 229, 224, 225, 220, 221, 216, 217, 212, 213, 208, 209, 204, 205, 200, 201, 196, 197, 192, 193]
]

#-----------------------------------------------------------------------
# load data
#-----------------------------------------------------------------------
def loadDelayScan(path):
    # hit rate per bit for every delay_<ns>.npy in path
    files = glob.glob(os.path.join(path, "delay_*.npy"))
    files = sorted(files, key=lambda x: float(re.search(r'delay_([-\d.]+)\.npy', x).group(1)))
    delays = np.array([float(re.search(r'delay_([-\d.]+)\.npy', f).group(1)) for f in files])
    hitRate, nSample = [], []
    for f in files:
        x = np.load(f) # (nsample, bit)
        hitRate.append(x.mean(0))
        nSample.append(x.shape[0])
    return delays, 100*np.array(hitRate), np.array(nSample)

def loadVthSweep(path):
    # hit rate per threshold, delay and bit for every vth<V> sub folder in path
    vthDirs = [d for d in glob.glob(os.path.join(path, "vth*")) if os.path.isdir(d)]
    vthDirs = sorted(vthDirs, key=lambda x: float(re.search(r'vth([\d.]+)$', x).group(1)))
    vths, delays, hitRate, nSample = [], [], [], []
    for d in vthDirs:
        de, hr, ns = loadDelayScan(d)
        if len(de) == 0:
            print(f"No delay_*.npy files in {d}. Skipping.")
            continue
        vths.append(1000*float(re.search(r'vth([\d.]+)$', d).group(1))) # mV
        delays.append(de)
        hitRate.append(hr)
        nSample.append(ns)
    if len(delays) == 0:
        return None
    # an aborted sweep can leave the last threshold with fewer delay points, keep the common ones
    nDelay = min(len(d) for d in delays)
    if any(len(d) != nDelay for d in delays):
        print(f"{path}: not all thresholds have the same number of delay points, keeping the first {nDelay}")
    return (np.array(vths), delays[0][:nDelay],
            np.stack([h[:nDelay] for h in hitRate], 0),   # (nVth, nDelay, bit) in %
            np.stack([n[:nDelay] for n in nSample], 0))   # (nVth, nDelay)

#-----------------------------------------------------------------------
# helpers
#-----------------------------------------------------------------------
def crossings(x, y, level):
    # first rising and last falling crossing of level, linear interpolation between neighbouring points
    above = y >= level
    if not above.any():
        return np.nan, np.nan
    iFirst = np.argmax(above)
    iLast = len(above) - 1 - np.argmax(above[::-1])
    rise = x[iFirst] if iFirst == 0 else np.interp(level, [y[iFirst-1], y[iFirst]], [x[iFirst-1], x[iFirst]])
    fall = x[iLast] if iLast == len(x)-1 else np.interp(level, [y[iLast+1], y[iLast]], [x[iLast+1], x[iLast]])
    return rise, fall

def widthAtHalfMaxVth(vths, delays, h, level):
    # h: (nVth, nDelay) hit rate in % for one bit
    # Vmax = highest threshold where the hit rate reaches level at some delay; width = delay range above level at Vmax/2
    reached = h.max(axis=1) >= level
    if not reached.any():
        return np.nan, np.nan, np.nan, False
    vmax = vths[reached].max()
    vhalf = vmax/2
    widths = np.array([np.subtract(*crossings(delays, row, level)[::-1]) for row in h])
    edge = np.array([row[0] >= level or row[-1] >= level for row in h]) # window touches the scan edge: width is a lower limit
    ok = np.isfinite(widths)
    if vhalf <= vths[ok].min():
        iV = np.argmin(vths[ok])
        return vmax, vhalf, widths[ok][iV], bool(edge[ok][iV])
    width = np.interp(vhalf, vths[ok], widths[ok])
    iNear = np.argsort(np.abs(vths - vhalf))[:2]
    return vmax, vhalf, width, bool(edge[iNear].any())

def binEdges(c):
    # bin edges centered on the scan points
    if len(c) == 1:
        return np.array([c[0]-0.5, c[0]+0.5])
    mid = (c[1:] + c[:-1])/2
    return np.concatenate([[c[0] - (mid[0]-c[0])], mid, [c[-1] + (c[-1]-mid[-1])]])

def labels(ax, nPix=None, y0=0.9, extra=None):
    SmartPixLabel(ax, 0.05, y0, size=22)
    ax.text(0.05, y0-0.05, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    txt = f"Pixel {int(nPix)}, " if nPix is not None else ""
    txt += f"V$_{{asic}}$ = {info['vasic']*1000:.0f} mV" if "vasic" in info else ""
    ax.text(0.05, y0-0.10, txt, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    if extra:
        ax.text(0.05, y0-0.15, extra, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

def save(fig, name):
    print(f"Saving file to {name}")
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

#-----------------------------------------------------------------------
# plots
#-----------------------------------------------------------------------
def plotHitRateVsDelay(delays, hitRate, hitRateErr, nPix, plotDir):
    # single delay scan: hit rate vs delay per bit
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel("Hit Rate [%]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        ax.errorbar(delays, hitRate[:,iB], yerr=hitRateErr[:,iB], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=3)
    ax.set_xlim(delays.min(), delays.max())
    ax.set_ylim(0, 160) # room for the labels
    legend = ax.legend(fontsize=15, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight('bold')
    SetTicks(ax)
    labels(ax, nPix, extra=f"Delay offset from {baseDly:.1f} ns" if baseDly is not None else None)
    save(fig, os.path.join(plotDir, f"PulseDelayScan_hitRate_Pixel{int(nPix)}.pdf"))

def plotHitMaps(vths, delays, hitRate, nPix, plotDir):
    # 2D hit map per bit: x = delay, y = threshold, z = hit rate
    for iB in range(NBIT):
        z = hitRate[:,:,iB]
        fig, ax = plt.subplots(figsize=(7,6))
        mesh = ax.pcolormesh(binEdges(delays), binEdges(vths), z, cmap=plt.cm.viridis, vmin=0, vmax=100)
        if len(vths) > 1 and len(delays) > 1 and z.min() < args.contour < z.max():
            ax.contour(delays, vths, z, levels=[args.contour], colors='white', linewidths=1.5)
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='top')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(f"Hit Rate, Bit {iB} [%] (white: {args.contour:.0f}%)", fontsize=18, labelpad=10)
        ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
        ax.set_ylabel(r"V$_{th}$ [mV]", fontsize=18, labelpad=10)
        SetTicks(ax)
        save(fig, os.path.join(plotDir, f"PulseDelayScan_hitMap_Bit{iB}.pdf"))

def plotCumulative(vths, hitRate, nPix, plotDir):
    # hit rate summed over all delays (fraction of all samples with a hit) vs threshold, per bit
    cumulative = hitRate.mean(axis=1) # (nVth, bit)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(r"V$_{th}$ [mV]", fontsize=18, labelpad=10)
    ax.set_ylabel("Cumulative Hit Rate, All Delays [%]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        ax.plot(vths, cumulative[:,iB], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=3)
    ax.set_xlim(vths.min(), vths.max())
    ax.set_ylim(0, max(1, cumulative.max()) * 1.6)
    legend = ax.legend(fontsize=15, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight('bold')
    SetTicks(ax)
    labels(ax, nPix)
    save(fig, os.path.join(plotDir, "PulseDelayScan_cumulativeHitRate.pdf"))
    return cumulative

def plotPixelMap(values, zlabel, name):
    # 8x32 pixel map in the MatrixNPix2D.py layout; pixels without a value are white
    arr = np.full((len(grid), len(grid[0])), np.nan)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            p = grid[i][j]
            if p in values and np.isfinite(values[p]):
                arr[len(grid)-1-i][j] = values[p]
    fig, ax = plt.subplots(figsize=(14,5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')
    mesh = ax.pcolormesh(np.arange(len(grid[0])+1), np.arange(len(grid)+1), np.ma.masked_invalid(arr), cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='top')
    cbar.set_label(zlabel, fontsize=18, labelpad=10)
    ax.set_xlabel('Column', fontsize=18, labelpad=10)
    ax.set_ylabel('Row', fontsize=18, labelpad=10)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            v = arr[len(grid)-1-i][j]
            ax.text(j + 0.5, len(grid)-1-i + 0.65, f"{grid[i][j]}", color="gray", ha="center", va="center", fontsize=6)
            if np.isfinite(v):
                ax.text(j + 0.5, len(grid)-1-i + 0.3, f"{v:.1f}", color="white", ha="center", va="center", fontsize=6)
    ax.set_xlim(0, len(grid[0]))
    ax.set_ylim(0, len(grid))
    save(fig, name)

#-----------------------------------------------------------------------
# run
#-----------------------------------------------------------------------
testType = info["testType"]

if testType == "PulseDelayScan":
    delays, hitRate, nSample = loadDelayScan(inPath)
    if len(delays) == 0:
        raise SystemExit(f"No data found in {inPath}")
    hitRateErr = np.sqrt(hitRate*(100-hitRate)/nSample[:,None])
    nPix = info.get("nPix", 0)
    np.savez(os.path.join(outDir, "pulse_delay_data.npz"), delay_ns=delays, hitRate=hitRate, hitRateErr=hitRateErr, nSample=nSample)
    plotHitRateVsDelay(delays, hitRate, hitRateErr, nPix, outDir)

elif testType == "PulseDelayScanVTH":
    data = loadVthSweep(inPath)
    if data is None:
        raise SystemExit(f"No data found in {inPath}")
    vths, delays, hitRate, nSample = data
    nPix = info.get("nPix", 0)
    plotHitMaps(vths, delays, hitRate, nPix, outDir)
    cumulative = plotCumulative(vths, hitRate, nPix, outDir)
    np.savez(os.path.join(outDir, "pulse_delay_data.npz"), delay_ns=delays, vth_mV=vths, hitRate=hitRate, cumulativeHitRate=cumulative, nSample=nSample, contour=args.contour)
    print(f"Data saved to {os.path.join(outDir, 'pulse_delay_data.npz')}")

elif testType == "PulseDelayScanMatrix":
    pixDirs = [d for d in glob.glob(os.path.join(inPath, "nPix*")) if os.path.isdir(d)]
    pixDirs = sorted(pixDirs, key=lambda x: int(re.search(r'nPix(\d+)$', x).group(1)))
    pixels, vmax, vhalf, width, atEdge = [], [], [], [], []
    hitRates, vthsAll, delaysAll = {}, None, None
    for d in pixDirs:
        nPix = int(re.search(r'nPix(\d+)$', d).group(1))
        data = loadVthSweep(d)
        if data is None:
            print(f"No data for pixel {nPix}. Skipping.")
            continue
        vths, delays, hitRate, nSample = data
        if not args.noPixelPlots:
            plotDir = os.path.join(outDir, f"nPix{nPix}")
            os.makedirs(plotDir, exist_ok=True)
            plotHitMaps(vths, delays, hitRate, nPix, plotDir)
            plotCumulative(vths, hitRate, nPix, plotDir)
        res = [widthAtHalfMaxVth(vths, delays, hitRate[:,:,iB], args.contour) for iB in range(NBIT)]
        pixels.append(nPix)
        vmax.append([r[0] for r in res]); vhalf.append([r[1] for r in res]); width.append([r[2] for r in res]); atEdge.append([r[3] for r in res])
        hitRates[nPix] = hitRate
        if vthsAll is None:
            vthsAll, delaysAll = vths, delays
        print(f"Pixel {nPix}: Vmax = {', '.join(f'{v:.1f}' for v in vmax[-1])} mV, width at Vmax/2 = {', '.join(f'{w:.1f}' for w in width[-1])} ns (bits 0, 1, 2)")

    if len(pixels) == 0:
        raise SystemExit(f"No data found in {inPath}")
    pixels, vmax, vhalf, width, atEdge = map(np.array, (pixels, vmax, vhalf, width, atEdge))
    if atEdge.any():
        print(f"WARNING: for {atEdge.sum()} pixel/bit combinations the hit window touches the edge of the delay scan at Vmax/2: the width is a lower limit. Widen the delay range.")

    # hit rates for all pixels with the same scan grid
    same = [p for p in pixels if hitRates[p].shape == hitRates[pixels[0]].shape]
    np.savez(os.path.join(outDir, "pulse_delay_matrix_data.npz"),
             nPix=pixels, vmax_mV=vmax, vhalf_mV=vhalf, widthAtHalfMaxVth_ns=width, widthAtScanEdge=atEdge,
             vth_mV=vthsAll, delay_ns=delaysAll, hitRate_nPix=np.array(same), hitRate=np.stack([hitRates[p] for p in same], 0), contour=args.contour)
    print(f"Data saved to {os.path.join(outDir, 'pulse_delay_matrix_data.npz')}")

    # summary: width at half maximum threshold, map and histogram per bit
    for iB in range(NBIT):
        plotPixelMap({p: w for p, w in zip(pixels, width[:,iB])}, f"Width at V$_{{th}}^{{max}}$/2, Bit {iB} [ns]", os.path.join(outDir, f"PulseDelayScan_widthAtHalfMaxVth_map_Bit{iB}.pdf"))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(r"Width at V$_{th}^{max}$/2 [ns]", fontsize=18, labelpad=10)
    ax.set_ylabel("Pixels", fontsize=18, labelpad=10)
    good = width[np.isfinite(width)]
    if len(good):
        bins = np.linspace(np.floor(good.min()) - 1, np.ceil(good.max()) + 1, 30)
        for iB in range(NBIT):
            w = width[:,iB][np.isfinite(width[:,iB])]
            ax.hist(w, bins=bins, histtype='step', color=color[iB], linewidth=2, label=f"Bit {iB}: mean {w.mean():.1f} ns, std {w.std():.1f} ns" if len(w) else f"Bit {iB}")
        ax.set_ylim(0, ax.get_ylim()[1] * 2.2) # room for the labels and legend
        legend = ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.03, 0.68), frameon=False)
    SetTicks(ax)
    labels(ax, extra=f"{len(pixels)} pixels")
    save(fig, os.path.join(outDir, "PulseDelayScan_widthAtHalfMaxVth_hist.pdf"))


else:
    raise SystemExit(f"Unknown test type {testType} for {inPath}")
