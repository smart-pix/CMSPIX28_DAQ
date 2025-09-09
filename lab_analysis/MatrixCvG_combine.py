import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import mplhep as hep
hep.style.use("ATLAS")

from SmartPixStyle import *
from Analyze import inspectPath

conf = {
    "SP1" : {
        "path" : "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID17_SuperPix1/2025.07.02_17.17.45_MatrixCvG_vMin0.001_vMax0.400_vStep0.00100_nSample1365.000_vdda0.900_BXCLKf10.00_BxCLKDly45.00_injDly75.00_Ibias0.600/plots/CvG_data.npy"
    },
    "SP2" : {
        "path" : "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID17_SuperPix2/2025.09.07_08.55.23_MatrixCvG_vMin0.001_vMax0.400_vStep0.00100_nSample1365.000_vdda0.900_BXCLKf10.00_BxCLKDly45.00_injDly75.00_Ibias0.600/plots/CvG_data.npy"
    }
}

# get information
info = inspectPath(os.path.dirname(conf["SP1"]["path"]))
print(info)


# plotting
fig, ax = plt.subplots(figsize=(6,6))
bins = np.linspace(45, 75, 16)

# loop and make the plot
for key, val in conf.items():
    x = np.load(val["path"])
    label = key 
    mu, std = norm.fit(x[:,2])
    label += (f" μ = {mu:.2f}, σ = {std:.2f}")
    ax.hist(x[:,2], bins=bins, label=label, histtype="step")

# make legend
legend = ax.legend(fontsize=15, loc = "upper right") #bbox_to_anchor=(0.03, 0.85), loc='upper left')
for text in legend.get_texts():
    text.set_fontweight('bold')

# axis labels
ax.set_xlabel("CvG [µV/e⁻]")
ax.set_ylabel("Count")

# label
SmartPixLabel(ax, 0, 1.0, text=f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, All Pixels and Bits", size=12, fontweight='normal', style='normal')
    
plt.savefig("Test.pdf")