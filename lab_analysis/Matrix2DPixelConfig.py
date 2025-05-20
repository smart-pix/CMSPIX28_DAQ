# python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse

# get the code from within subroutines
import sys
SpacelySubroutinesPath = os.path.join(os.path.dirname(os.getcwd()), "spacely/PySpacely/spacely-asic-config/CMSPIX28Spacely")
sys.path.append(SpacelySubroutinesPath)
from CMSPIX28Spacely_Subroutines_A0 import genPixelConfigFromInputCSV, genPixelProgramList, grid
filename = os.path.join(SpacelySubroutinesPath, "csv/compouts.csv")
pixelLists, pixelValues = genPixelConfigFromInputCSV(filename)

# style
import mplhep as hep
hep.style.use("ATLAS")
# local imports
from SmartPixStyle import *

# conver the ROIC 8x32 grid to the sensor 16x16 grid
def convert_8x32_to_16x16(output):
    rows, cols = output.shape
    assert rows == 8 and cols == 32, "Input must be 8x32"

    new_rows = 16
    new_cols = 16
    result = np.zeros((new_rows, new_cols), dtype=output.dtype)

    for i in range(rows):
        # even columns from output row i → becomes row 2*i in result
        
        result[2*i, :] = output[i, 0::2]  # columns 1,3,5,... (odd indices) 
        # odd columns from output row i → becomes row 2*i + 1 in result
        
        result[2*i + 1, :] = output[i, 1::2]
        
    return result

def doPlot(output_array, outDir, drawPixelNumbers=False):

    # interpret grid type
    gridtype = "ROIC" if output_array.shape[0] == 8 else "SENSOR"

    # plot    
    fig, ax = plt.subplots(figsize=(6, 6))

    # Base colormap
    base_cmap = plt.cm.viridis
    # Sample 3 colors from viridis at 1/3, 2/3, and 1
    positions = [1/3, 2/3, 1.0]
    viridis_colors = base_cmap(positions)
    # Create list: white + those 3 sampled colors
    colors = [(1, 1, 1, 1)] + [tuple(c) for c in viridis_colors]
    cmap = ListedColormap(colors)
    
    # boundaries
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # plot the 2D histogram
    binsx = np.linspace(0, output_array.shape[0], output_array.shape[0]+1)
    binsy = np.linspace(0, output_array.shape[1], output_array.shape[1]+1)
    h2dx = np.tile(np.arange(output_array.shape[1]),output_array.shape[0])
    h2dy = np.repeat(np.arange(output_array.shape[0]), output_array.shape[1])
    cax = ax.hist2d(
        h2dx, 
        h2dy, 
        bins=[binsy, binsx], 
        weights=output_array.flatten(), 
        cmap=cmap,
        norm=norm
    )

    # set the colorbar
    cbar = fig.colorbar(cax[3], ax=ax, orientation='horizontal', location='top', ticks=[0, 1, 2, 3])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("Thermometric pixel value", fontsize=18, labelpad=10)
    ax.set_xlabel('Column', fontsize=18, labelpad=10)
    ax.set_ylabel('Row', fontsize=18, labelpad=10)
    ax.grid(which='both', color='black', linestyle='-', linewidth=1, alpha=1)
    
    ax.set_xlim(0, output_array.shape[1]) # len(grid[0]))
    ax.set_ylim(0, output_array.shape[0]) # len(grid))
    
    # set ticks
    # SetTicks(ax)
    ax.set_xticks(np.arange(0, output_array.shape[1]+1, 4 if gridtype == "ROIC" else 2))
    ax.set_yticks(np.arange(0, output_array.shape[0]+1, 2))
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=16))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=1, direction="in", pad=8)
    
    # put pixel number inside the grid
    if drawPixelNumbers and gridtype == "ROIC":
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                # color = "white" if output_array[i][j] > 0 else "black"
                ax.text(j + 0.5, len(grid)-1-i + 0.5, f"{grid[i][j]}", color="gray", ha="center", va="center", fontsize=5)
                ax.text(j + 0.5, len(grid)-1-i + 0.3, f"{output_array[len(grid)-1-i][j]:.1f}", color="gray", ha="center", va="center", fontsize=2)

    # save fig
    outFileName = os.path.join(outDir, f"Matrix2DPixelConfig_{gridtype}_Pattern{iP}.pdf")
    print(f"Saving file to {outFileName}")
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--inPatternIndices", type=int, nargs='+', help='One or pattern indexes. Default will do only pattern 0. User can pass in multiple, ex. 1 2 3. If user passes in -1 then all patterns are drawn', default=[0])
parser.add_argument("-o", '--outDir', type=str, default="./", help='Output path')
args = parser.parse_args()

# interpret the input
patternIndices = args.inPatternIndices
if patternIndices[0] == -1:
    patternIndices = list(range(len(pixelLists)))

# loop over pattern indices
for iP in patternIndices:

    # pick the pixel programming
    index_array = np.array(pixelLists[iP])
    data_values = np.array(pixelValues[iP])

    # Step 1: Create output matrix of zeros with same shape
    output = np.zeros_like(grid, dtype=data_values.dtype)

    # Step 2: Find row, col indices in the grid for each index
    positions = np.argwhere(np.isin(grid, index_array))

    # Step 3: Map each index in index_array to its corresponding position
    # Build a dict from grid values to their (row, col)
    value_to_pos = {val: tuple(np.argwhere(grid == val)[0]) for val in index_array}

    # Step 4: Assign data_values to those positions
    for val, data in zip(index_array, data_values):
        row, col = value_to_pos[val]
        output[row, col] = data

    # make output array
    output_array = np.array(output) # 8x32 style
    output_array16 = convert_8x32_to_16x16(output_array) # 16x16 style

    # save the output
    doPlot(output_array, args.outDir)
    doPlot(output_array16, args.outDir)
