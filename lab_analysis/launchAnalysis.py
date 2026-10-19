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
    args = parser.parse_args()

    # # Glob pattern for matching folders
    # base_path = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2"
    # pattern = f"{base_path}/2025.05.01_13*"
    # # Find all matching folders
    # folders = sorted(glob.glob(pattern))

    folders = sorted(glob.glob(args.inPath))

    # A "linearity" run folder (SCurveSweepVTHPix's shared root) nests results as
    # nPix#/vth# instead of the old flat per-pixel top-level folder. Expand each such
    # root into its nPix# subfolders so the rest of this script still processes one
    # pixel per folder, exactly like the old flat MatrixVTH layout. Remember which
    # roots got expanded so we can combine their per-pixel results afterward.
    # Gated on "linearity": MatrixNPix and MatrixCvG runs also have nPix* children,
    # but those are handled as a single run by Analyze.py and must not be expanded.
    expanded_folders = []
    linearityRoots = []
    for f in folders:
        nPixSubfolders = sorted(glob.glob(os.path.join(f, "nPix*"))) if "linearity" in os.path.basename(f) else []
        if nPixSubfolders:
            expanded_folders.extend(nPixSubfolders)
            linearityRoots.append(f)
        else:
            expanded_folders.append(f)
    folders = expanded_folders

    # Loop over folders and run commands
    for folder in folders:
        print(f"Processing folder: {folder}")
        info = inspectPath(folder)

        try:
            Analyze = ["python", "Analyze.py", "-i", folder]
            if args.doFit:
                Analyze.append("--doFit")
            subprocess.run(Analyze, check=True)
            subprocess.run(["python", "SCurve.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixCalibration" in folder:
                subprocess.run(["python", "SCurveFWCal.py", "-i", folder, "--combine"], check=True)
            if info.get("testType") == "Single" and info.get("nPix") is None:
                subprocess.run(["python", "MatrixNPix.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
                # subprocess.run(["python", "MatrixNPix2D.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixNPix" in folder:
                subprocess.run(["python", "MatrixNPix.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
                subprocess.run(["python", "MatrixNPix2D.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixVTH" in folder or "linearity" in folder:
                subprocess.run(["python", "MatrixVTH.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if  "MatrixIbias" in folder:
                subprocess.run(["python", "MatrixIbias.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixInjDly" in folder or "MatrixBxCLKDly" in folder:
                subprocess.run(["python", "MatrixInjDly.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixPulseGenFall" in folder:
                subprocess.run(["python", "MatrixPulseGenFall.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixCvG" in folder:
                subprocess.run(["python", "MatrixCvG.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {folder}: {e}")

    # Each "linearity" root's 256 nPix# folders were just analyzed individually above
    # (one scurve_data.npz per pixel). MatrixCvG.py needs all pixels combined into a
    # single features array to produce the per-pixel CvG/vth-offset fits and their
    # mu/sigma histograms (slope and offset, per bit), so combine and run it here.
    for root in linearityRoots:
        try:
            subprocess.run(["python", "combineLinearityPixels.py", "-i", root], check=True)
            subprocess.run(["python", "MatrixCvG.py", "-i", os.path.join(root, "plots/scurve_data.npz")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error combining/running MatrixCvG for {root}: {e}")
