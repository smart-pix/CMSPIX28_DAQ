import glob
import subprocess
import argparse
import os

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
    
    # Loop over folders and run commands
    for folder in folders:
        print(f"Processing folder: {folder}")
        try:
            Analyze = ["python", "Analyze.py", "-i", folder]
            if args.doFit:
                Analyze.append("--doFit")
            subprocess.run(Analyze, check=True)
            subprocess.run(["python", "SCurve.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixCalibration" in folder:
                subprocess.run(["python", "SCurveFWCal.py", "-i", folder, "--combine"], check=True)
            if "MatrixNPix" in folder:
                subprocess.run(["python", "MatrixNPix.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
                subprocess.run(["python", "MatrixNPix2D.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixVTH" in folder:
                subprocess.run(["python", "MatrixVTH.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixInjDly" in folder:
                subprocess.run(["python", "MatrixInjDly.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixPulseGenFall" in folder:
                subprocess.run(["python", "MatrixPulseGenFall.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {folder}: {e}")
