import numpy as np
import os
import glob
import re
import json
import argparse
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("ATLAS")

from SmartPixStyle import *
from scipy.optimize import curve_fit
from scipy.stats import norm
from Analyze import inspectPath, NBIT, addUnitArgs, xAxisUnit, VtomV

# Analysis of the pulse generator delay scans (CMSPIX28Spacely_Subroutines_B6_DelayScan.py)
# Input is the test folder:
#   PulseDelayScan         : <folder>/delay_<ns>.npy
#   PulseDelayScanVTH      : <folder>/vth<V>/delay_<ns>.npy
#   PulseDelayScanMatrix   : <folder>/nPix<n>/vth<V>/delay_<ns>.npy
#   ScurveDelaySweep       : <folder>/delay_<ns>/vth_<V>.npy          (threshold swept at every delay, same plots as PulseDelayScanVTH)
#   ScurveDelaySweepMatrix : <folder>/nPix<n>/delay_<ns>/vth_<V>.npy  (same plots as PulseDelayScanMatrix)
#   ScurveDelaySweepISource: <folder>/ISource<I>uA/delay_<ns>/vth_<V>.npy  (one pixel, several ISource10uA currents)
#   ScurveDelaySweepAmp    : <folder>/delay_<ns>/vasic_<V>.npy        (fixed threshold, pulse generator amplitude swept at every delay)
# The ScurveDelaySweep plots are named ScurveDelaySweep_* instead of PulseDelayScan_*
# Each npy has shape (nsample, 3) with the 3 bits of the pixel for every sample
# Outputs in <folder>/plots (or -o):
#   PulseDelayScan       : PulseDelayScan_hitRate_Pixel<n>.pdf (hit rate vs delay per bit), pulse_delay_data.npz
#   PulseDelayScanVTH    : PulseDelayScan_hitMap_Bit<b>.pdf (2D hit map delay vs threshold per bit)
#                          PulseDelayScan_cumulativeHitRate.pdf (hit rate over all delays vs threshold per bit)
#                          pulse_delay_data.npz
#   PulseDelayScanMatrix : the same two plots per pixel in nPix<n>/ (skip with --noPixelPlots)
#                          PulseDelayScan_widthAtHalfMaxVth_map_Bit<b>.pdf (pixel map) and _hist.pdf
#                          pulse_delay_matrix_data.npz
#   --noise (matrix, no injected charge): fit of the hit rate over all delays vs threshold with a falling Gaussian S-curve per pixel and bit,
#                          hit rate = 100 * norm.sf(Vth, mu, sigma): mu = baseline [mV] (pedestal + comparator offset), sigma = noise [mV]
#                          {prefix}_noise_{mu,sigma}_map_Bit<b>.pdf, _hist.pdf (bits overlaid), _bitDiff_hist.pdf, _matrixSCurve.pdf, noise_matrix_data.npz
#                          (replaces the width at half maximum threshold summary)
#   ScurveDelaySweepISource : the 2D maps and cumulative hit rate per current in ISource<I>uA/
#                             ScurveDelaySweep_vmax_vs_ISource.pdf, ScurveDelaySweep_widthAtHalfMaxVth_vs_ISource.pdf (per bit)
#                             pulse_delay_isource_data.npz
#   ScurveDelaySweepAmp  : ScurveDelaySweepAmp_hitMap_Bit<b>.pdf (2D hit map delay vs V_asic per bit)
#                          ScurveDelaySweepAmp_vasic50_vs_delay.pdf (lowest V_asic with >= --contour hit rate at every delay, per bit)
#                          ScurveDelaySweepAmp_window_vs_vasic.pdf (first and last delay with >= --contour hit rate vs V_asic, per bit: time walk)
#                          pulse_delay_amp_data.npz
# Timing overlay on the 2D maps (needs --injOffset), from a scope measurement with the run's settings, all relative to the scanLoad rising edge:
#   --injOffset : injection edge at the chip at delay offset 0      --anaRise : BxCLK_ANA rising edge      --bxclkRise : BxCLK rising edge
#   an event at time t after the scanLoad rising edge is at delay offset t - injOffset on the maps
#   Drawn: sampling edge = BxCLK_ANA rising edge (white dashed), BxCLK rising edge (red dotted), scanLoad rise / fall (orange)
#   The hit window is set by the pixel sampling its comparator shortly after the BxCLK_ANA rising edge: the late edge of the window is
#   that sampling instant minus the time walk, the early edge is set by the time over threshold
# Width at half maximum threshold (matrix only), per pixel and bit:
#   maximum threshold Vmax = highest threshold at which the hit rate reaches 50% at some delay
#   width = delay range [ns] with hit rate >= 50% at Vmax/2 (interpolated between the scanned thresholds)

# Argument parser
parser = argparse.ArgumentParser(description='Analyze pulse generator delay scans.')
parser.add_argument("-i", '--inPath', type=str, required=True, help='Input test folder')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Output directory. Default is <inPath>/plots')
parser.add_argument('--contour', type=float, default=50, help='Hit rate [%%] used for the 2D map contour and the width at half maximum threshold')
parser.add_argument('--noise', action="store_true", help='Matrix scans without injected charge: fit the noise S-curve (baseline and sigma) of every pixel and bit instead of the width summary')
parser.add_argument('--noPixelPlots', action="store_true", help='PulseDelayScanMatrix: only make the matrix summary plots, not the plots for every pixel')
parser.add_argument('--injOffset', type=float, default=None, help='Scope: time [ns] from the scanLoad rising edge to the injection edge at the chip at delay offset 0. Enables the timing overlay on the 2D maps')
parser.add_argument('--anaRise', type=float, default=None, help='Scope: time [ns] from the scanLoad rising edge to the BxCLK_ANA rising edge (sampling edge)')
parser.add_argument('--bxclkRise', type=float, default=None, help='Scope: time [ns] from the scanLoad rising edge to the BxCLK rising edge')
parser.add_argument('--bxclkPeriod', type=str, default='28', help='bxclk_period (hex) if the run has no scan_settings.json, for the scanLoad falling edge')
addUnitArgs(parser) # --cvg (per bit CvG fits) or --cvg <uV/e-> [--vthOffset <mV>]: threshold axis in electrons
args = parser.parse_args()

# threshold axis: mV, or electrons through the pixel gain, per bit (see addUnitArgs in Analyze.py)
vthScale, vthOffset, vthUnit = xAxisUnit(args, vthAxis=True)
vthLabel = f"Threshold [{vthUnit}]" if vthUnit != "mV" else r"V$_{th}$ [mV]"
def vthShow(vths, iB):
    # thresholds in mV to the shown unit for bit iB
    return (np.asarray(vths)/VtomV - vthOffset[iB]) * vthScale[iB]

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

def loadScurveDelaySweep(path, var="vth"):
    # hit rate per threshold (var = "vth") or amplitude (var = "vasic"), delay and bit for every delay_<ns> sub folder with <var>_<V>.npy files in path
    dlyDirs = [d for d in glob.glob(os.path.join(path, "delay_*")) if os.path.isdir(d)]
    dlyDirs = sorted(dlyDirs, key=lambda x: float(re.search(r'delay_([-\d.]+)$', x).group(1)))
    vths, delays, hitRate, nSample = [], [], [], []
    for d in dlyDirs:
        files = glob.glob(os.path.join(d, f"{var}_*.npy"))
        if len(files) == 0:
            print(f"No {var}_*.npy files in {d}. Skipping.")
            continue
        pattern = var + r'_([\d.]+)\.npy'
        files = sorted(files, key=lambda x: float(re.search(pattern, x).group(1)))
        x = [np.load(f) for f in files] # (nsample, bit) per threshold
        vths.append(np.array([1000*float(re.search(pattern, f).group(1)) for f in files])) # mV
        delays.append(float(re.search(r'delay_([-\d.]+)$', d).group(1)))
        hitRate.append(100*np.array([i.mean(0) for i in x]))
        nSample.append(np.array([i.shape[0] for i in x]))
    if len(delays) == 0:
        return None
    # an aborted sweep can leave the last delay with fewer thresholds, keep the common ones
    nVth = min(len(v) for v in vths)
    if any(len(v) != nVth for v in vths):
        print(f"{path}: not all delays have the same number of {var} points, keeping the first {nVth}")
    return (vths[0][:nVth], np.array(delays),
            np.stack([h[:nVth] for h in hitRate], 1),   # (nVth, nDelay, bit) in %
            np.stack([n[:nVth] for n in nSample], 1))   # (nVth, nDelay)

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
    txt = txt.rstrip(", ")
    ax.text(0.05, y0-0.10, txt, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    if extra:
        ax.text(0.05, y0-0.15, extra, transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')

def save(fig, name):
    print(f"Saving file to {name}")
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

def runSettings(path):
    # timing settings of the run: scan_settings.json written by the routines, else the command line options
    found = sorted(glob.glob(os.path.join(path, "**", "scan_settings.json"), recursive=True))
    if found:
        with open(found[0]) as fs:
            return json.load(fs)
    return {"bxclk_period": args.bxclkPeriod}

def timingOverlay(settings, injOffset, anaRise, bxclkRise):
    # scope-measured edges (ns after the scanLoad rising edge) converted to delay offset coordinates
    if injOffset is None:
        return None
    period = 2.5*int(settings["bxclk_period"], 16)  # scanLoad stays high for one BxCLK period
    toDelay = lambda t: None if t is None else t - injOffset
    print(f"Timing overlay (delay offset): scanLoad high {toDelay(0):.1f} to {toDelay(period):.1f} ns"
          + (f", sampling edge (BxCLK_ANA rising) {toDelay(anaRise):.1f} ns" if anaRise is not None else ", no --anaRise: no sampling edge")
          + (f", BxCLK rising {toDelay(bxclkRise):.1f} ns" if bxclkRise is not None else ""))
    return {"scanLoad": (toDelay(0), toDelay(period)), "sampling": toDelay(anaRise), "bxclk": toDelay(bxclkRise)}

def drawTiming(ax, overlay, xmin, xmax):
    if overlay is None:
        return
    first = True
    for x in overlay["scanLoad"]:
        if xmin <= x <= xmax:
            ax.axvline(x, color="orange", linestyle="-", linewidth=1.2, label="scanLoad rise / fall" if first else None)
            first = False
    if overlay["sampling"] is not None and xmin <= overlay["sampling"] <= xmax:
        ax.axvline(overlay["sampling"], color="white", linestyle="--", linewidth=1.8, label="sampling edge (BxCLK_ANA rising)")
    if overlay["bxclk"] is not None and xmin <= overlay["bxclk"] <= xmax:
        ax.axvline(overlay["bxclk"], color="red", linestyle=":", linewidth=1.8, label="BxCLK rising")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)

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
    vthsMV = vths
    for iB in range(NBIT):
        vths = vthShow(vthsMV, iB)
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
        ax.set_ylabel(vthLabel, fontsize=18, labelpad=10)
        SetTicks(ax)
        edges = binEdges(delays)
        drawTiming(ax, overlay, edges[0], edges[-1])
        ax.set_xlim(edges[0], edges[-1])
        save(fig, os.path.join(plotDir, f"{prefix}_hitMap_Bit{iB}.pdf"))

def plotCumulative(vths, hitRate, nPix, plotDir):
    # hit rate summed over all delays (fraction of all samples with a hit) vs threshold, per bit
    cumulative = hitRate.mean(axis=1) # (nVth, bit)
    vthsBit = [vthShow(vths, iB) for iB in range(NBIT)] # the conversion to electrons can differ per bit
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(vthLabel, fontsize=18, labelpad=10)
    ax.set_ylabel("Cumulative Hit Rate, All Delays [%]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        ax.plot(vthsBit[iB], cumulative[:,iB], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=3)
    ax.set_xlim(min(v.min() for v in vthsBit), max(v.max() for v in vthsBit))
    ax.set_ylim(0, max(1, cumulative.max()) * 1.6)
    legend = ax.legend(fontsize=15, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight('bold')
    SetTicks(ax)
    labels(ax, nPix)
    save(fig, os.path.join(plotDir, f"{prefix}_cumulativeHitRate.pdf"))
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

def fitNoise(vths, h):
    # h: hit rate [%] vs threshold [mV] for one bit, no injected charge
    # falling Gaussian S-curve: hit rate = 100 * P(noise + baseline > Vth) -> mu = baseline, sigma = noise (both mV)
    # fit only if the curve falls from >= 80% to <= 20% inside the scan, else nan
    if not (h.max() >= 80 and h[-1] <= 20):
        return np.nan, np.nan, np.nan
    i50 = np.argmax(h < 50)
    v50 = np.interp(50, [h[i50], h[i50-1]], [vths[i50], vths[i50-1]]) if i50 > 0 else vths[0]
    try:
        (mu, sigma), _ = curve_fit(lambda x, m, sg: 100*norm.sf(x, m, sg), vths, h, p0=[v50, max(0.5, (vths[-1]-vths[0])/20)], bounds=((-np.inf, 1e-3), (np.inf, np.inf)))
    except Exception:
        return v50, np.nan, np.nan
    return v50, mu, sigma

def noiseSummary(pixels, noiseCurves):
    # noise characterization of the matrix: baseline mu and noise sigma per pixel and bit
    res = {p: [fitNoise(*(lambda v, h: (v, h[:,iB]))(*noiseCurves[p])) for iB in range(NBIT)] for p in pixels}
    v50 = np.array([[r[0] for r in res[p]] for p in pixels])
    mu = np.array([[r[1] for r in res[p]] for p in pixels])
    sigma = np.array([[r[2] for r in res[p]] for p in pixels])
    for p, m, sg in zip(pixels, mu, sigma):
        print(f"Pixel {p}: baseline = {', '.join(f'{x:.2f}' for x in m)} mV, noise = {', '.join(f'{x:.2f}' for x in sg)} mV (bits 0, 1, 2)")
    nBad = np.isnan(sigma).sum()
    if nBad:
        print(f"WARNING: {nBad} pixel/bit combinations have no noise fit (S-curve not falling from >= 80% to <= 20% inside the threshold range). Widen the threshold range.")
    np.savez(os.path.join(outDir, "noise_matrix_data.npz"), nPix=pixels, v50_mV=v50, mu_mV=mu, sigma_mV=sigma,
             vth_mV=noiseCurves[pixels[0]][0], hitRate=np.stack([noiseCurves[p][1] for p in pixels if noiseCurves[p][1].shape == noiseCurves[pixels[0]][1].shape], 0))
    print(f"Data saved to {os.path.join(outDir, 'noise_matrix_data.npz')}")

    for values, name, zlabel, xlab in [(mu, "mu", "Baseline", r"Baseline $\mu$ [mV]"), (sigma, "sigma", "Noise", r"Noise $\sigma$ [mV]")]:
        # pixel map per bit
        for iB in range(NBIT):
            plotPixelMap({p: v for p, v in zip(pixels, values[:,iB])}, f"{zlabel}, Bit {iB} [mV]", os.path.join(outDir, f"{prefix}_noise_{name}_map_Bit{iB}.pdf"))
        # histogram, bits overlaid
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlabel(xlab, fontsize=18, labelpad=10)
        ax.set_ylabel("Pixels", fontsize=18, labelpad=10)
        good = values[np.isfinite(values)]
        if len(good):
            bins = np.linspace(good.min() - 0.05*np.ptp(good) - 0.1, good.max() + 0.05*np.ptp(good) + 0.1, 40)
            for iB in range(NBIT):
                v = values[:,iB][np.isfinite(values[:,iB])]
                ax.hist(v, bins=bins, histtype='step', color=color[iB], linewidth=2, label=f"Bit {iB}: mean {v.mean():.2f}, std {v.std():.2f} mV" if len(v) else f"Bit {iB}")
            ax.set_ylim(0, ax.get_ylim()[1] * 2.2) # room for the labels and legend
            ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.03, 0.68), frameon=False)
        SetTicks(ax)
        labels(ax, extra=f"{len(pixels)} pixels, no injected charge")
        save(fig, os.path.join(outDir, f"{prefix}_noise_{name}_hist.pdf"))

    # bit by bit: difference to bit 0 for the same pixel
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(r"Bit $-$ Bit 0, same pixel [mV]", fontsize=18, labelpad=10)
    ax.set_ylabel("Pixels", fontsize=18, labelpad=10)
    diffs = [(f"$\\mu$, bit {iB}", mu[:,iB] - mu[:,0], "-", color[iB]) for iB in range(1, NBIT)] + [(f"$\\sigma$, bit {iB}", sigma[:,iB] - sigma[:,0], "--", color[iB]) for iB in range(1, NBIT)]
    good = np.concatenate([d[np.isfinite(d)] for _, d, _, _ in diffs])
    if len(good):
        bins = np.linspace(good.min() - 0.1, good.max() + 0.1, 40)
        for lab, d, ls, c in diffs:
            d = d[np.isfinite(d)]
            if len(d):
                ax.hist(d, bins=bins, histtype='step', color=c, linestyle=ls, linewidth=2, label=f"{lab}: mean {d.mean():.2f}, std {d.std():.2f} mV")
        ax.set_ylim(0, ax.get_ylim()[1] * 2.2)
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.03, 0.68), frameon=False)
    SetTicks(ax)
    labels(ax, extra=f"{len(pixels)} pixels, no injected charge")
    save(fig, os.path.join(outDir, f"{prefix}_noise_bitDiff_hist.pdf"))

    # matrix average noise S-curve per bit
    vths = noiseCurves[pixels[0]][0]
    curves = np.stack([noiseCurves[p][1] for p in pixels if noiseCurves[p][1].shape == noiseCurves[pixels[0]][1].shape], 0) # (pixel, nVth, bit)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(vthLabel, fontsize=18, labelpad=10)
    ax.set_ylabel("Hit Rate, No Injection [%]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        v = vthShow(vths, iB)
        m = curves[:,:,iB].mean(0)
        ax.plot(v, m, color=color[iB], marker='o', markersize=3, label=f"Bit {iB} (mean of {len(curves)} pixels)")
    ax.set_ylim(0, 160) # room for the labels
    ax.legend(fontsize=12, loc="center right")
    SetTicks(ax)
    labels(ax)
    save(fig, os.path.join(outDir, f"{prefix}_noise_matrixSCurve.pdf"))

#-----------------------------------------------------------------------
# run
#-----------------------------------------------------------------------
testType = info["testType"]
settings = runSettings(inPath)
# FW mode: the delay is the pulse generator offset plus the firmware injection_delay steps, not the pulse generator delay alone
if settings.get("delayMode") == "FW" or "_dlyModeFW" in inPath:
    xlabel = "Delay Offset [ns]"
mapTypes = ("PulseDelayScanVTH", "PulseDelayScanMatrix", "ScurveDelaySweep", "ScurveDelaySweepMatrix", "ScurveDelaySweepISource", "ScurveDelaySweepAmp")
# ScurveDelaySweep: threshold swept at every delay, same plots with their own names
prefix = "ScurveDelaySweep" if testType.startswith("ScurveDelaySweep") else "PulseDelayScan"
if testType == "ScurveDelaySweepAmp":
    prefix = "ScurveDelaySweepAmp"
loadMap = loadScurveDelaySweep if testType.startswith("ScurveDelaySweep") else loadVthSweep
overlay = timingOverlay(settings, args.injOffset, args.anaRise, args.bxclkRise) if testType in mapTypes else None
if overlay is None and testType in mapTypes:
    print("No --injOffset given: 2D maps without the timing overlay")

if testType == "PulseDelayScan":
    delays, hitRate, nSample = loadDelayScan(inPath)
    if len(delays) == 0:
        raise SystemExit(f"No data found in {inPath}")
    hitRateErr = np.sqrt(hitRate*(100-hitRate)/nSample[:,None])
    nPix = info.get("nPix", 0)
    np.savez(os.path.join(outDir, "pulse_delay_data.npz"), delay_ns=delays, hitRate=hitRate, hitRateErr=hitRateErr, nSample=nSample)
    plotHitRateVsDelay(delays, hitRate, hitRateErr, nPix, outDir)

elif testType in ("PulseDelayScanVTH", "ScurveDelaySweep"):
    data = loadMap(inPath)
    if data is None:
        raise SystemExit(f"No data found in {inPath}")
    vths, delays, hitRate, nSample = data
    nPix = info.get("nPix", 0)
    plotHitMaps(vths, delays, hitRate, nPix, outDir)
    cumulative = plotCumulative(vths, hitRate, nPix, outDir)
    np.savez(os.path.join(outDir, "pulse_delay_data.npz"), delay_ns=delays, vth_mV=vths, hitRate=hitRate, cumulativeHitRate=cumulative, nSample=nSample, contour=args.contour)
    print(f"Data saved to {os.path.join(outDir, 'pulse_delay_data.npz')}")

elif testType in ("PulseDelayScanMatrix", "ScurveDelaySweepMatrix"):
    pixDirs = [d for d in glob.glob(os.path.join(inPath, "nPix*")) if os.path.isdir(d)]
    pixDirs = sorted(pixDirs, key=lambda x: int(re.search(r'nPix(\d+)$', x).group(1)))
    pixels, vmax, vhalf, width, atEdge = [], [], [], [], []
    hitRates, vthsAll, delaysAll, noiseCurves = {}, None, None, {}
    for d in pixDirs:
        nPix = int(re.search(r'nPix(\d+)$', d).group(1))
        data = loadMap(d)
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
        if args.noise:
            noiseCurves[nPix] = (vths, hitRate.mean(axis=1)) # hit rate over all delays (nVth, bit)
        if vthsAll is None:
            vthsAll, delaysAll = vths, delays
        if not args.noise:
            print(f"Pixel {nPix}: Vmax = {', '.join(f'{v:.1f}' for v in vmax[-1])} mV, width at Vmax/2 = {', '.join(f'{w:.1f}' for w in width[-1])} ns (bits 0, 1, 2)")

    if len(pixels) == 0:
        raise SystemExit(f"No data found in {inPath}")
    pixels, vmax, vhalf, width, atEdge = map(np.array, (pixels, vmax, vhalf, width, atEdge))
    if args.noise:
        noiseSummary(pixels, noiseCurves)
        raise SystemExit(0)
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
        plotPixelMap({p: w for p, w in zip(pixels, width[:,iB])}, f"Width at V$_{{th}}^{{max}}$/2, Bit {iB} [ns]", os.path.join(outDir, f"{prefix}_widthAtHalfMaxVth_map_Bit{iB}.pdf"))
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
    save(fig, os.path.join(outDir, f"{prefix}_widthAtHalfMaxVth_hist.pdf"))

elif testType == "ScurveDelaySweepISource":
    curDirs = [d for d in glob.glob(os.path.join(inPath, "ISource*uA")) if os.path.isdir(d)]
    curDirs = sorted(curDirs, key=lambda x: float(re.search(r'ISource([-\d.]+)uA$', x).group(1)))
    nPix = info.get("nPix", 0)
    currents, vmax, width, atEdge, hitRates = [], [], [], [], []
    for d in curDirs:
        cur = float(re.search(r'ISource([-\d.]+)uA$', d).group(1))
        data = loadMap(d)
        if data is None:
            print(f"No data for ISource10uA = {cur:.2f} uA. Skipping.")
            continue
        vths, delays, hitRate, nSample = data
        plotDir = os.path.join(outDir, os.path.basename(d))
        os.makedirs(plotDir, exist_ok=True)
        plotHitMaps(vths, delays, hitRate, nPix, plotDir)
        plotCumulative(vths, hitRate, nPix, plotDir)
        res = [widthAtHalfMaxVth(vths, delays, hitRate[:,:,iB], args.contour) for iB in range(NBIT)]
        currents.append(cur)
        vmax.append([r[0] for r in res]); width.append([r[2] for r in res]); atEdge.append([r[3] for r in res])
        hitRates.append(hitRate)
        print(f"ISource10uA = {cur:.2f} uA: Vmax = {', '.join(f'{v:.1f}' for v in vmax[-1])} mV, width at Vmax/2 = {', '.join(f'{w:.1f}' for w in width[-1])} ns (bits 0, 1, 2)")

    if len(currents) == 0:
        raise SystemExit(f"No data found in {inPath}")
    currents, vmax, width, atEdge = map(np.array, (currents, vmax, width, atEdge))
    if atEdge.any():
        print(f"WARNING: for {atEdge.sum()} current/bit combinations the hit window touches the edge of the delay scan at Vmax/2: the width is a lower limit. Widen the delay range.")
    same = all(h.shape == hitRates[0].shape for h in hitRates)
    np.savez(os.path.join(outDir, "pulse_delay_isource_data.npz"), isource_uA=currents, vmax_mV=vmax, widthAtHalfMaxVth_ns=width, widthAtScanEdge=atEdge,
             vth_mV=vths, delay_ns=delays, hitRate=np.stack(hitRates, 0) if same else np.array([]), contour=args.contour)
    print(f"Data saved to {os.path.join(outDir, 'pulse_delay_isource_data.npz')}")

    # summary: maximum threshold and width at half maximum threshold vs current, per bit
    for values, ylabel, name in [(vmax, r"V$_{th}^{max}$ [mV]", "vmax"), (width, r"Width at V$_{th}^{max}$/2 [ns]", "widthAtHalfMaxVth")]:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlabel(r"ISource10uA [$\mu$A]", fontsize=18, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=18, labelpad=10)
        for iB in range(NBIT):
            ax.plot(currents, values[:,iB], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=5)
        good = values[np.isfinite(values)]
        if len(good):
            ax.set_ylim(0, good.max() * 1.6) # room for the labels
        legend = ax.legend(fontsize=15, loc="upper right")
        for text in legend.get_texts():
            text.set_fontweight('bold')
        SetTicks(ax)
        labels(ax, nPix)
        save(fig, os.path.join(outDir, f"{prefix}_{name}_vs_ISource.pdf"))

elif testType == "ScurveDelaySweepAmp":
    data = loadScurveDelaySweep(inPath, var="vasic")
    if data is None:
        raise SystemExit(f"No data found in {inPath}")
    amps, delays, hitRate, nSample = data # amps: V_asic [mV], hitRate (nAmp, nDelay, bit) in %
    nPix = info.get("nPix", 0)
    vthInfo = f"V$_{{th}}$ = {info['vth']*1000:.0f} mV" if info.get("vth") is not None else None
    ampLabel = r"V$_{asic}$ [mV]"

    # 2D hit map per bit: x = delay, y = amplitude, z = hit rate
    for iB in range(NBIT):
        z = hitRate[:,:,iB]
        fig, ax = plt.subplots(figsize=(7,6))
        mesh = ax.pcolormesh(binEdges(delays), binEdges(amps), z, cmap=plt.cm.viridis, vmin=0, vmax=100)
        if len(amps) > 1 and len(delays) > 1 and z.min() < args.contour < z.max():
            ax.contour(delays, amps, z, levels=[args.contour], colors='white', linewidths=1.5)
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='top')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(f"Hit Rate, Bit {iB} [%] (white: {args.contour:.0f}%)", fontsize=18, labelpad=10)
        ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
        ax.set_ylabel(ampLabel, fontsize=18, labelpad=10)
        SetTicks(ax)
        edges = binEdges(delays)
        drawTiming(ax, overlay, edges[0], edges[-1])
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(binEdges(amps)[0], binEdges(amps)[-1])
        save(fig, os.path.join(outDir, f"{prefix}_hitMap_Bit{iB}.pdf"))

    # lowest amplitude reaching the contour hit rate at every delay (interpolated), per bit
    amp50 = np.array([[crossings(amps, hitRate[:,iD,iB], args.contour)[0] for iB in range(NBIT)] for iD in range(len(delays))]) # (nDelay, bit)
    # first and last delay with the contour hit rate at every amplitude, per bit (the window edges: time walk)
    window = np.array([[crossings(delays, hitRate[iA,:,iB], args.contour) for iB in range(NBIT)] for iA in range(len(amps))]) # (nAmp, bit, rise/fall)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(f"V$_{{asic}}$ at {args.contour:.0f}% Hit Rate [mV]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        ax.plot(delays, amp50[:,iB], label=f'Bit {iB}', color=color[iB], linestyle='-', marker='o', markersize=3)
    good = amp50[np.isfinite(amp50)]
    ax.set_xlim(delays.min(), delays.max())
    if len(good):
        ax.set_ylim(0, good.max() * 1.6) # room for the labels
    legend = ax.legend(fontsize=15, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight('bold')
    SetTicks(ax)
    labels(ax, nPix, extra=vthInfo)
    save(fig, os.path.join(outDir, f"{prefix}_vasic50_vs_delay.pdf"))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(ampLabel, fontsize=18, labelpad=10)
    ax.set_ylabel(f"Hit Window Edge at {args.contour:.0f}% [ns]", fontsize=18, labelpad=10)
    for iB in range(NBIT):
        ax.plot(amps, window[:,iB,0], color=color[iB], linestyle='-', marker='o', markersize=3, label=f'Bit {iB} first')
        ax.plot(amps, window[:,iB,1], color=color[iB], linestyle='--', marker='s', markersize=3, label=f'Bit {iB} last')
    ax.set_xlim(amps.min(), amps.max())
    ax.set_ylim(delays.min(), delays.max() + 1.2*(delays.max() - delays.min())) # room for the labels
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(0.98, 0.72), ncol=2, frameon=False)
    SetTicks(ax)
    labels(ax, nPix, extra=vthInfo)
    save(fig, os.path.join(outDir, f"{prefix}_window_vs_vasic.pdf"))

    np.savez(os.path.join(outDir, "pulse_delay_amp_data.npz"), delay_ns=delays, vasic_mV=amps, hitRate=hitRate, nSample=nSample,
             vasic50_mV=amp50, windowFirst_ns=window[:,:,0], windowLast_ns=window[:,:,1], vth_mV=1000*info["vth"] if info.get("vth") is not None else np.nan, contour=args.contour)
    print(f"Data saved to {os.path.join(outDir, 'pulse_delay_amp_data.npz')}")

else:
    raise SystemExit(f"Unknown test type {testType} for {inPath}")
