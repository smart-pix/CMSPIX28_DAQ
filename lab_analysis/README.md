To run dynamic_parseDNNresults.py (used to find and/or parse on-chip NN results): you have to pass the location to the readout.csv (output file from on-chip NN), and one of the following:
1. -d flag: if dnn_RTL_out.csv is produced and copied to the the same location of readout.csv. Passing this flag find the best timestamp to evaluate the on-chip NN results at. It will also produce the relevant confusion matrix and accuracy of on-chip NN w.r.t. RTL results.
2. -s flag followed by timestamp number: if you want to evaluate the on-chip NN results at a particular timestamp
If both of the above flags, it will evaluate the on-chip NN results at the user-defined timestamp and compare the subsequent results with RTL prediction.

Analysis of CvG data (across all 256 pixels) to obtain conversion equations to take us from VTH [V] to VTH [electrons]:
Note: Future iteration of CvG routine will improve upon directory management. Please bear with us for now with the below defined fix.

1. Folder organization: Ensure the directory is named in the usual format (date_MatrixCvG_vMin_vMax...). Within this directory needs to be folders: nPix0, nPix1, ... nPix254, nPix256. And within each nPix folder, the data for various vth's used needs to be present.
2. Running analysis: python3 launchAnalysis.py -i data_MatrixCvG_vMin_vMax... (--doFit if you want SCurves to be fit with Gaussian)
3. Final plots: python3 MatrixCvG_allPixels.py -i data_MatrixCvG_vMin_vMax.../plots/scurve_data.npz
One can use a shell script like the following to help move directories:
```
master="./2025.11.20_11.24.21_MatrixCvG_vMin0.001_vMax0.400_vStep0.00100_nSample1365.000_vdda0.900_BXCLKf10.00_BxCLKDly45.00_injDly75.00_Ibias0.600"
list="dirs.txt"

while IFS= read -r d; do
    [ -z "$d" ] && continue        # skip empty lines
    mv -- "$d" "$master"/
done < "$list"

cd "$master"

for d in *_nPix*; do
    # extract from first occurrence of "nPix" to end
    new="${d#*nPix}"
    new="nPix${new}"          # ensure it starts with nPix
    mv -- "$d" "$new"
done
```

To perform analysis on power measurements: pass the location of the Ivddd.csv file using the -f flag. Subsequent arguments are to help understand the double-peaks typically observed in power consumption histograms/values. The -c argument is the user's guess value of the power value that separates the two peaks. The -r argument sets the maximum range of power-histogram plot in the case of P_noise measurements.