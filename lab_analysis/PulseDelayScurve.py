import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

try:
    import mplhep as hep
    hep.style.use("ATLAS")
except ImportError:
    print("mplhep not available, using default matplotlib style")

from SmartPixStyle import *

# Extract the S-curve (hit rate vs threshold) at a given pulse delay from the
# pulse_delay_data.npz produced by PulseDelayScan.py for a PulseDelayScanVTH test
# (ROUTINE_PulseDelayScanSweepVTH).
# Example:
#   python PulseDelayScurve.py -i <folder>/plots/pulse_delay_data.npz -d 12
#   python PulseDelayScurve.py -i <folder> -d -8 -d 0 -d 12     (several delays, the folder's plots/ is used)
# The nearest scanned delay is used. Outputs <outDir>/PulseDelayScurve_delay<d>ns.pdf and .npz
# The fit is hitRate = A/2 * erfc((vth - mu)/(sqrt(2) sigma)) (falling S-curve), per bit.

parser = argparse.ArgumentParser(description='Extract the S-curve at a given delay from a PulseDelayScanVTH npz file.')
parser.add_argument("-i", '--inPath', type=str, required=True, help='pulse_delay_data.npz file, or the test folder containing plots/pulse_delay_data.npz')
parser.add_argument("-d", '--delay', type=float, action="append", help='Delay [ns] to extract the S-curve at (can be repeated). If omitted you are prompted.')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Output directory. Default is the directory of the npz file')
parser.add_argument('--noFit', action="store_true", help='Do not fit the S-curves')
args = parser.parse_args()

inFile = os.path.normpath(args.inPath)
if os.path.isdir(inFile):
    inFile = os.path.join(inFile, "plots", "pulse_delay_data.npz")
outDir = args.outDir if args.outDir else os.path.dirname(inFile)
os.makedirs(outDir, exist_ok=True)

d = np.load(inFile)
for k in ("delay_ns", "vth_mV", "hitRate", "nSample"):
    if k not in d.files:
        raise SystemExit(f"{inFile} has no '{k}': it must come from a PulseDelayScanVTH analysis (keys: {d.files})")
delays, vths, hitRate, nSample = d["delay_ns"], d["vth_mV"], d["hitRate"], d["nSample"]  # hitRate: (nVth, nDelay, bit) in %

delayReq = args.delay
if not delayReq:
    delayReq = [float(input(f"Delay [ns] in [{delays.min():.1f}, {delays.max():.1f}]: "))]

color = ["blue", "red", "orange"]
chipInfo = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(inFile))))  # ChipVersion*_ChipID*_SuperPix*

def scurve(x, A, mu, sigma):
    return A/2 * erfc((x - mu)/(np.sqrt(2)*abs(sigma)))

def fit(x, y, err):
    # returns (popt, perr) or (None, None)
    if y.max() <= 0:
        return None, None
    p0 = [y.max(), x[np.argmin(abs(y - y.max()/2))], max((x[-1]-x[0])/20, 0.5)]
    try:
        popt, pcov = curve_fit(scurve, x, y, p0=p0, sigma=np.where(err > 0, err, 1.0), absolute_sigma=True, maxfev=10000)
        return popt, np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"  fit failed: {e}")
        return None, None

for req in delayReq:
    iD = int(np.argmin(abs(delays - req)))
    dly = delays[iD]
    if abs(dly - req) > 0.5*np.median(np.diff(delays)) + 1e-9:
        print(f"Warning: requested {req} ns is outside the scan [{delays.min()}, {delays.max()}]")
    print(f"Requested delay {req} ns -> using {dly:.2f} ns")

    hr = hitRate[:, iD, :]                                     # (nVth, bit) in %
    n = nSample[:, iD]
    err = np.sqrt(np.clip(hr*(100-hr), 0, None)/n[:, None])    # binomial error in %
    err = np.maximum(err, 100/(2*n[:, None]))                  # avoid zero error at 0% / 100%

    fig, ax = plt.subplots(figsize=(8, 6))
    fitRes = np.full((hr.shape[1], 3, 2), np.nan)              # bit, (A, mu, sigma), (value, err)
    for iB in range(hr.shape[1]):
        ax.errorbar(vths, hr[:, iB], yerr=err[:, iB], label=f'Bit {iB}', color=color[iB], linestyle='none', marker='o', markersize=3)
        if not args.noFit:
            popt, perr = fit(vths, hr[:, iB], err[:, iB])
            if popt is not None:
                fitRes[iB, :, 0], fitRes[iB, :, 1] = popt, perr
                xs = np.linspace(vths.min(), vths.max(), 500)
                ax.plot(xs, scurve(xs, *popt), color=color[iB], linestyle='-',
                        label=f'$\\mu$ = {popt[1]:.1f} mV, $\\sigma$ = {abs(popt[2]):.1f} mV')
                print(f"  Bit {iB}: A = {popt[0]:.1f}%, mu = {popt[1]:.2f} +- {perr[1]:.2f} mV, sigma = {abs(popt[2]):.2f} +- {perr[2]:.2f} mV")
    ax.set_xlabel("Threshold [mV]")
    ax.set_ylabel("Hit rate [%]")
    ax.set_ylim(0, 115)
    SmartPixLabel(ax, 0.05, 0.9, size=22)
    ax.text(0.05, 0.85, f"{chipInfo}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    ax.text(0.05, 0.80, f"Pulse delay = {dly:.1f} ns", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    ax.legend(loc="upper right", fontsize=12)

    name = f"PulseDelayScurve_delay{dly:.1f}ns"
    for ext in ("pdf", "png"):
        print(f"Saving file to {os.path.join(outDir, name)}.{ext}")
        fig.savefig(os.path.join(outDir, f"{name}.{ext}"), bbox_inches='tight')
    plt.close(fig)
    np.savez(os.path.join(outDir, f"{name}.npz"), delay_ns=dly, vth_mV=vths, hitRate=hr, hitRateErr=err, nSample=n,
             fit_A_mu_sigma=fitRes[:, :, 0], fit_A_mu_sigma_err=fitRes[:, :, 1])
