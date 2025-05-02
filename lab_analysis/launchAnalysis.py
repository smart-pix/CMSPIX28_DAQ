import glob
import subprocess
import argparse

if __name__ == "__main__":

    # user arguments
    parser = argparse.ArgumentParser(description='Producing simple histograms.')
    parser.add_argument('-i', '--inPath', required=True, help='Path to input files')
    # parser.add_argument('-j', '--ncpu', type=int, default=1, help='Number of CPUs to use')
    # parser.add_argument('-o', '--outDir', default=None, help="Output directory. If not provided then use directory of inFilePath")
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
            subprocess.run(["python", "Analyze.py", "-i", folder], check=True)
            subprocess.run(["python", "SCurveFWCal.py", "-i", folder], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {folder}: {e}")
