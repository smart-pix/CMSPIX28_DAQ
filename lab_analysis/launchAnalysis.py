import glob
import subprocess
import argparse
import os
from Analyze import inspectPath

if __name__ == "__main__":

    # user arguments
    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inPath', required=True, help='Path to input files')
    # parser.add_argument('-j', '--ncpu', type=int, default=1, help='Number of CPUs to use')
    # parser.add_argument('-o', '--outDir', default=None, help="Output directory. If not provided then use directory of inFilePath")
    parser.add_argument('--doFit', action="store_true", help="Run the S-cruve fitting. Note the analysis will take significantly longer.")
    parser.add_argument('--staticPulse', action="store_true", help="Data from PreProgSCurveBurst_StaticPulse (fixed pulse amplitude, Vth swept)")
    parser.add_argument('--electrons', action="store_true", help="Show the S-curve axis / threshold in electrons instead of mV (injection: Q = V*Cin/q; Vth axes need --cvg)")
    parser.add_argument('--cvg', '--CvG', nargs='?', const="fits", default=None, help="Vth axes (StaticPulse S-curves, PulseDelayScan / ScurveDelaySweep maps) in electrons: alone, with the per bit CvG fits in Analyze.py; or a pixel gain [uV/e-] for all bits")
    parser.add_argument('--vthOffset', type=float, default=None, help="Vth [mV] at 0 e-, used with a --cvg value")
    parser.add_argument('--injOffset', type=float, default=None, help="PulseDelayScan: scope time [ns] from scanLoad rising edge to injection at the chip at delay offset 0; enables the timing overlay on the 2D maps")
    parser.add_argument('--anaRise', type=float, default=None, help="PulseDelayScan: scope time [ns] from scanLoad rising edge to the BxCLK_ANA rising edge (sampling edge)")
    parser.add_argument('--bxclkRise', type=float, default=None, help="PulseDelayScan: scope time [ns] from scanLoad rising edge to the BxCLK rising edge")
    args = parser.parse_args()

    # # Glob pattern for matching folders
    # base_path = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2"
    # pattern = f"{base_path}/2025.05.01_13*"
    # # Find all matching folders
    # folders = sorted(glob.glob(pattern))
    #print("test_1")
    folders = sorted(glob.glob(args.inPath))

    # unit options for the plotting scripts (addUnitArgs in Analyze.py)
    unitOpts = ["--electrons"] if args.electrons else []
    if args.cvg is not None:
        unitOpts += ["--cvg"] if args.cvg == "fits" else ["--cvg", args.cvg]
    if args.vthOffset is not None:
        unitOpts += ["--vthOffset", str(args.vthOffset)]
    scurveOpts = unitOpts + (["--staticPulse"] if args.staticPulse else [])

    # Loop over folders and run commands
    for folder in folders:
        print(f"Processing folder: {folder}")
        info = inspectPath(folder)
        #print("test_2")
        # pulse generator delay scans (PulseDelayScan*, ScurveDelaySweep*) have their own analysis, no S-curve fit
        if "PulseDelayScan" in folder or "ScurveDelaySweep" in folder:
            try:
                cmd = ["python", "PulseDelayScan.py", "-i", folder]
                for opt in ["injOffset", "anaRise", "bxclkRise"]:
                    if getattr(args, opt) is not None:
                        cmd += [f"--{opt}", str(getattr(args, opt))]
                subprocess.run(cmd + unitOpts, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {folder}: {e}")
            continue
        try:
            Analyze = ["python", "Analyze.py", "-i", folder]
            #print("test_3")
            if args.doFit:
                Analyze.append("--doFit")
            subprocess.run(Analyze, check=True)
            subprocess.run(["python", "SCurve.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
            if "MatrixCalibration" in folder:
                subprocess.run(["python", "SCurveFWCal.py", "-i", folder, "--combine"], check=True)
            if info.get("testType") == "Single" and info.get("nPix") is None:
                subprocess.run(["python", "MatrixNPix.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
                # subprocess.run(["python", "MatrixNPix2D.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixNPix" in folder:
                subprocess.run(["python", "MatrixNPix.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
                subprocess.run(["python", "MatrixNPix2D.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
            if "MatrixVTH" in folder:
                subprocess.run(["python", "MatrixVTH.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
            if  "MatrixIbias" in folder:
                subprocess.run(["python", "MatrixIbias.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
            if "MatrixInjDly" in folder or "MatrixBxCLKDly" in folder:
                subprocess.run(["python", "MatrixInjDly.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
            if "MatrixPulseGenFall" in folder:
                subprocess.run(["python", "MatrixPulseGenFall.py", "-i", os.path.join(folder, "plots/scurve_data.npz")] + scurveOpts, check=True)
            if "MatrixCvG" in folder:
                # MatrixCvG_allPixels.py handles the volt-valued S-curve axis and the static-pulse
                # layout; MatrixCvG.py does not and would overwrite the same histogram files
                CvG = ["python", "MatrixCvG_allPixels.py", "-i", os.path.join(folder, "plots/scurve_data.npz")]
                if args.staticPulse:
                    CvG.append("--staticPulse")
                subprocess.run(CvG, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {folder}: {e}")
