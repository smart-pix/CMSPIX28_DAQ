#!/bin/bash

base_path="/Users/bparpillon/Library/CloudStorage/OneDrive-FermiNationalAcceleratorLaboratory/CMStest/data/ChipVersion1_ChipID9_SuperPix2/2025.04.22_14.59.18_MatrixCalibration_vMin0.001_vMax0.400_vStep0.03400_nSample50.000_vdda0.900_VTH0.800_BXCLK10.00_BxCLK_DELAY_0x12_SCAN_INJ_DLY_0x1D_SCAN_LOAD_DLY_0x13_scanLoadPhase_0x26"

for i in {0..255}; do
    folder="${base_path}/nPix$i"
    if [ -d "$folder" ]; then
        echo "Running on $folder"
        python3 Analyze.py -i "$folder"
    else
        echo "Skipping missing folder: $folder"
    fi
done