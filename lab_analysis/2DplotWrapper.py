from matplotlib.backends.backend_pdf import PdfPages
from SCurveFWCal import generate_plot
import os
import glob
import matplotlib.pyplot as plt

base_path = "/Users/bparpillon/Library/CloudStorage/OneDrive-FermiNationalAcceleratorLaboratory/CMStest/data/ChipVersion1_ChipID9_SuperPix2/2025.04.22_14.59.18_MatrixCalibration_vMin0.001_vMax0.400_vStep0.03400_nSample50.000_vdda0.900_VTH0.800_BXCLK10.00_BxCLK_DELAY_0x12_SCAN_INJ_DLY_0x1D_SCAN_LOAD_DLY_0x13_scanLoadPhase_0x26/"  # Replace with the correct base path
pdf_output = "AllPixPlots.pdf"

with PdfPages(pdf_output) as pdf:
    for i in range(3):
        # Define the path to the 'plots' subfolder
        plots_folder = os.path.join(base_path, f"nPix{i}", "plots")
        
        # Check if the 'plots' folder exists
        if not os.path.isdir(plots_folder):
            print(f"Skipping folder: {plots_folder} (folder doesn't exist)")
            continue
        
        # Find the .npz file within the 'plots/' subfolder
        npz_files = glob.glob(os.path.join(plots_folder, "*.npz"))
        
        if not npz_files:
            print(f"No .npz file found in {plots_folder}")
            continue
        
        # Define the path to 'settings.npy' (parent folder of 'plots')
        settings_file = os.path.join(base_path, f"nPix{i}", "settings.npy")
        print(f"Settings file path: {settings_file}")  # Debugging line
        if not os.path.exists(settings_file):
            print(f"Warning: {settings_file} not found. Skipping this folder.")
            continue  # Skip this folder if settings.npy is missing
        
        file_path = npz_files[0]  # Use the first .npz file found
        print(f"Processing file: {file_path}")  # Debugging line

        try:
            fig = generate_plot(file_path, settings_file)  # Pass both npz file and settings file
            if fig:
                pdf.savefig(fig)
                print(f"added plot for nPix{i} to pdf")
                plt.close(fig)
            else:
                print(f"Warning: generate_plot returned None for {file_path}.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")