import numpy as np
import argparse
import matplotlib.pyplot as plt
import os 
import mplhep as hep

hep.style.use("ATLAS")

# local imports
from SmartPixStyle import *
from Analyze import inspectPath

grid = [
    [63,  62,  59,  58,  55,  54,  51,  50,  47,  46,  43,  42,  39,  38,  35,  34,  31,  30,  27,  26,  23,  22,  19,  18,  15,  14,  11,  10,   7,   6,   3,    2],
    [60,  61,  56,  57,  52,  53,  48,  49,  44,  45,  40,  41,  36,  37,  32,  33,  28,  29,  24,  25,  20,  21,  16,  17,  12,  13,   8,   9,   4,   5,   0,    1],
    [127, 126, 123, 122, 119, 118, 115, 114, 111, 110, 107, 106, 103, 102, 99,  98,  95,  94,  91,  90,  87,  86,  83,  82,  79,  78,  75,  74,  71,  70,  67,   66],
    [124, 125, 120, 121, 116, 117, 112, 113, 108, 109, 104, 105, 100, 101, 96,  97,  92,  93,  88,  89,  84,  85,  80,  81,  76,  77,  72,  73,  68,  69,  64,   65],
    [191, 190, 187, 186, 183, 182, 179, 178, 175, 174, 171, 170, 167, 166, 163, 162, 159, 158, 155, 154, 151, 150, 147, 146, 143, 142, 139, 138, 135, 134, 131, 130],
    [188, 189, 184, 185, 180, 181, 176, 177, 172, 173, 168, 169, 164, 165, 160, 161, 156, 157, 152, 153, 148, 149, 144, 145, 140, 141, 136, 137, 132, 133, 128, 129],
    [255, 254, 251, 250, 247, 246, 243, 242, 239, 238, 235, 234, 231, 230, 227, 226, 223, 222, 219, 218, 215, 214, 211, 210, 207, 206, 203, 202, 199, 198, 195, 194],
    [252, 253, 248, 249, 244, 245, 240, 241, 236, 237, 232, 233, 228, 229, 224, 225, 220, 221, 216, 217, 212, 213, 208, 209, 204, 205, 200, 201, 196, 197, 192, 193]
]

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", '--inFilePath', type=str, required=True, help='Input file path')
parser.add_argument("-o", '--outDir', type=str, default=None, help='Output path')
args = parser.parse_args()

# input file
inData = np.load(args.inFilePath)
features = inData["features"]
# expect that features is of shape (nsetting, npix, nbit, vasic step)
# expect that for MatrixNPix test that there is only one setting
print(features.shape)
# features = features[0]

# get information
info = inspectPath(os.path.dirname(args.inFilePath))
print(info)

# output directory
outDir = args.outDir if args.outDir else os.path.dirname(args.inFilePath)
os.makedirs(outDir, exist_ok=True)

pltConfig = []
pltConfig.append({
    "name" : "nelectron_asic_50perc",
    "zlabel": r"S-Curve Half Max [e$^{-}$]", 
    "zlim" : [[0, 2000], [800, 3000], [2000, 4500]]
})
pltConfig.append({
    "name" : "scurve_mean",
    "zlabel": r"S-Curve $\mu$ [e$^{-}$]", 
    "zlim" : [[0, 2000], [800, 3000], [2000, 4500]]
})
pltConfig.append({
    "name" : "scurve_std",
    "zlabel": r"S-Curve $\sigma$ [e$^{-}$]", 
    "zlim" : [[0, 500], [0, 500], [0, 500]]
})

iB = 0
# for iS in range(features.shape[0]):
#     for iB in range(3):

#         # get data to plot
#         bit = features[iS:,:,iB:,]
#         print(bit.shape)
#         if np.all(bit==-999):
#             print(f"All values are -999 for feature {config['idx']}. Skipping plotting.")
#             continue
for iS in range(features.shape[0]):
    for iB in range(3):

        bit = features[iS:,:,iB]
        print(bit.shape)
        bit = np.squeeze(bit, axis=0) # remove the setting axis
        print(bit.shape)
        bit = bit[bit[:, 0].argsort()]
        
        # print out two values so the user can check by eye
        print(bit[bit[:, 0] == 63], bit[bit[:, 0] == 193])
    
        # 1=50perc, 2=mean, 3=std
        for fIdx in range(1, 4):  
            
            if np.all(bit[:,fIdx]==-999):
                print(f"All values are -999 for feature {fIdx}. Skipping plotting.")
                continue

            # Create an empty 2D array with the same shape as grid but populated with fIdx
            output_array = np.zeros((len(grid), len(grid[0])))
        
            # Fill the output_array with the corresponding values from bit0
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    pixel_number = grid[i][j]
                    output_array[len(grid)-1-i][j] = bit[bit[:, 0] == pixel_number, fIdx] # set this so that the order is correct top to bottom len(grid)-1-i
        
            # if value is <0 then bit failed so put -np.nan
            output_array[output_array<0] = -np.nan

            # plot
            fig, ax = plt.subplots()
            
            # plot the 2D histogram
            cmap = plt.cm.viridis
            cmap.set_bad(color='white')
            binsx = np.linspace(0, output_array.shape[0], output_array.shape[0]+1)
            binsy = np.linspace(0, output_array.shape[1], output_array.shape[1]+1)
            h2dx = np.tile(np.arange(32),8)
            h2dy = np.repeat(np.arange(8), 32)
            cax = ax.hist2d(
                h2dx, 
                h2dy, 
                bins=[binsy, binsx], 
                weights=output_array.flatten(), 
                cmap=cmap,
                vmin=pltConfig[fIdx-1]["zlim"][iB][0], 
                vmax=pltConfig[fIdx-1]["zlim"][iB][1]
            )
            
            # set the colorbar
            cbar = fig.colorbar(cax[3], ax=ax, orientation='horizontal', location='top')
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.set_label(pltConfig[fIdx - 1]["zlabel"], fontsize=18, labelpad=10)
            ax.set_xlabel('Column', fontsize=18, labelpad=10)
            ax.set_ylabel('Row', fontsize=18, labelpad=10)
            ax.grid(which='both', color='black', linestyle='-', linewidth=1, alpha=1)
            
            ax.set_xlim(0, len(grid[0]))
            ax.set_ylim(0, len(grid))
            
            # set ticks
            # SetTicks(ax)
            
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=16))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=1, direction="in", pad=8)
            
            # put pixel number inside the grid
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    # color = "white" if output_array[i][j] > 0 else "black"
                    ax.text(j + 0.5, len(grid)-1-i + 0.5, f"{grid[i][j]}", color="gray", ha="center", va="center", fontsize=6)
                    ax.text(j + 0.5, len(grid)-1-i + 0.3, f"{output_array[len(grid)-1-i][j]:.1f}", color="gray", ha="center", va="center", fontsize=3)
                    
            # save fig
            outFileName = os.path.join(outDir, f"grid_{pltConfig[fIdx - 1]['name']}_array_Setting{iS}_Bit{iB}.pdf")
            print(f"Saving file to {outFileName}")
            plt.savefig(outFileName, bbox_inches='tight')
            plt.close()



