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
    parser.add_argument('--staticPulse', action="store_true", help="MatrixCvG data from PreProgSCurveBurst_StaticPulse (fixed pulse amplitude, Vth swept)")
    args = parser.parse_args()

    # # Glob pattern for matching folders
    # base_path = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2"
    # pattern = f"{base_path}/2025.05.01_13*"
    # # Find all matching folders
    # folders = sorted(glob.glob(pattern))
    #print("test_1")
    folders = sorted(glob.glob(args.inPath))
    
    # Loop over folders and run commands
    for folder in folders:
        print(f"Processing folder: {folder}")
        info = inspectPath(folder)
        #print("test_2")
        # pulse generator delay scans have delay_*.npy instead of vasic_*.npy, no S-curve analysis
        if "PulseDelayScan" in folder:
            try:
                subprocess.run(["python", "PulseDelayScan.py", "-i", folder], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {folder}: {e}")
            continue
        try:
            Analyze = ["python", "Analyze.py", "-i", folder]
            #print("test_3")
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
            if "MatrixVTH" in folder:
                subprocess.run(["python", "MatrixVTH.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if  "MatrixIbias" in folder:
                subprocess.run(["python", "MatrixIbias.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixInjDly" in folder or "MatrixBxCLKDly" in folder:
                subprocess.run(["python", "MatrixInjDly.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixPulseGenFall" in folder:
                subprocess.run(["python", "MatrixPulseGenFall.py", "-i", os.path.join(folder, "plots/scurve_data.npz")], check=True)
            if "MatrixCvG" in folder:
                # MatrixCvG_allPixels.py handles the volt-valued S-curve axis and the static-pulse
                # layout; MatrixCvG.py does not and would overwrite the same histogram files
                CvG = ["python", "MatrixCvG_allPixels.py", "-i", os.path.join(folder, "plots/scurve_data.npz")]
                if args.staticPulse:
                    CvG.append("--staticPulse")
                subprocess.run(CvG, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {folder}: {e}")
