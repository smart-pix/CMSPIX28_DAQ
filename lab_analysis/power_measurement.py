import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress
import argparse
import sys, os
from SmartPixStyle import *
import mplhep as hep
from Analyze import *
hep.style.use("ATLAS")


parser = argparse.ArgumentParser(description='Extract data and fit with Gaussian.')
parser.add_argument('-f', type=str, help='Path to the CSV file containing the data.')
parser.add_argument('-c', '--compout', type=str, help='Path to the compout CSV file.', default='/asic/projects/C/CMS_PIX_28/dshekar/filter/model_pipeline/tmp_16x16_realisticClustInjection/1000_1600_2400/compouts_ylocal_0.00_1.35.csv')
parser.add_argument('-r', type=int, default=1000, help='Maximum range of power values in plot.')
parser.add_argument('-d', type=int, default=None, help='Debug double peak if passed. Value passed as argument will be chosen as the cut between the two peaks.')
args = parser.parse_args()

def get_y_cluster_size(event_matrix):
    # event_matrix: 2D numpy array of shape (16, 16)
    # Returns number of rows containing at least one nonzero
    return np.count_nonzero(event_matrix.sum(axis=1))

# Read the CSV file from the provided file path
file_path = os.path.join(args.f, '')
csv_file = file_path + 'Ivddd.csv'
df = pd.read_csv(csv_file)

Ivddd_post = df["Ivddd_post"]
Ivddd_pre = df["Ivddd_pre"]
Vvddd = df["Vvddd"]
# DNN_power was updated on 17Dec25 by DS to pass the last n_tb test vectors instead of the (previous default method of sending the) first n_tb test vectors. The following couple of lines takes into account this change.
if "EventNumber" in df.columns:
    event_number = df["EventNumber"]
    print("Event numbers read from CSV file.")
else:
    event_number = np.arange(len(Ivddd_post))
    print("Event numbers not found in CSV file, using default sequential numbering: 0 - n_tb.")
if "Noise" not in csv_file:
    Ivddd = (df["Ivddd_post"] - df["Ivddd_pre"]) * 100
    plot_xmin, plot_xmax = 0, args.r  # Set the x-axis range
    num_bins = 50
else:
    Ivddd = df["Ivddd_post"] * 100
    plot_xmin, plot_xmax = 3000, 4850  # Set the x-axis range
    num_bins = 175

plt.figure(figsize=(8, 6))
plt.scatter(range(len(Ivddd_pre)), Ivddd_pre*100, color='blue', alpha=0.7, label='Ivddd_pre')
plt.xlabel('Event number')
plt.ylabel('Ivddd_pre [A]')
plt.grid()
plt.title('Scatterplot of Ivddd_pre')
plt.legend()
plt.savefig(f'{file_path}IvdddPre_vs_event.png')

Pvddd = Ivddd * 0.9 * 1000  # Calculate power in uW

# plt.hist(Pvddd, bins=num_bins, range=(plot_xmin, plot_xmax), alpha=0.6, color='g', label="Pvddd net histogram")

mean = np.mean(Pvddd)
rms = np.sqrt(np.mean(np.square(Pvddd - mean)))
fit_min = mean - rms * 3
fit_max = mean + rms * 3
Pvddd_filtered = Pvddd[(Pvddd >= fit_min) & (Pvddd <= fit_max)]
mu, std = norm.fit(Pvddd_filtered)

error_on_mean = std / np.sqrt(len(Pvddd_filtered))  # Error on fit mean
error_on_stat_mean = rms / np.sqrt(len(Pvddd))  # Error on statistical mean

fig, ax = plt.subplots(figsize=(8,6))
plt.xlim(plot_xmin, plot_xmax)
counts, bins, _ = plt.hist(Pvddd, bins=num_bins, range=(plot_xmin, plot_xmax), facecolor='none', edgecolor='black', linewidth=1.5, label="Data")
bin_width = bins[1] - bins[0]  # Width of each bin
scaling_factor = len(Pvddd) * bin_width  # Total area under the histogram
x = np.linspace(plot_xmin, plot_xmax, 100)
p = norm.pdf(x, mu, std) * scaling_factor  # Scale the PDF to match the histogram

plt.plot(x, p, color='r', linewidth=2, label=f"Gaussian Fit\n$\mu={mu:.2e}, \sigma={std:.2e}$", alpha=0.5)
# plt.title("Histogram of power values with Gaussian Fit")
plt.xlabel("Power [$\mu$W]", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid()
plt.legend(loc='upper right', fontsize=14)
# plt.text(0.95, 0.80,
#          f"Fit mean = {mu:.2e} ± {error_on_mean:.2e} $\mu$W\n"
#          f"Stat. mean = {mean:.2e} ± {error_on_stat_mean:.2e} $\mu$W\n"
#          f"Stat. std. dev. = {rms:.2e} $\mu$W", 
#          transform=plt.gca().transAxes, fontsize=10, 
#          verticalalignment='top', horizontalalignment='right')
         
SmartPixLabel(ax, 0, 1.003, size=18)
info = inspectPath(file_path)
print("Chip ID and super pixel ID extracted:", int(info['ChipID']), ", ", int(info['SuperPix']))
ax.text(1, 1.005, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='right', va='bottom') #TODO
plt.savefig(f'{file_path}power_hist.pdf', bbox_inches='tight')

event_rows = 16
event_cols = 16
pixels_per_event = event_rows * event_cols
compout_file_path = args.compout
print("Reading compout file from: ", compout_file_path)

cluster_sizes = []

with open(compout_file_path, 'r') as f:
    lines = f.readlines()
    for idx in event_number:
        line = lines[idx]
        line = line.strip()
        if not line:
            continue
        pixels = [int(val) for val in line.split(',')]
        matrix = np.array(pixels).reshape(16, 16)
        y_cluster_size = np.count_nonzero(matrix.sum(axis=1))
        cluster_sizes.append(y_cluster_size)
    print("total events processed:", len(cluster_sizes))
    # print(f"Event {event_idx+1}: y-cluster size = {y_cluster_size}")

assert cluster_sizes == cluster_sizes[:len(Pvddd)], "Number of cluster sizes calculated from event number list is different from number of power values."

# Scatter plot of power vs. cluster size
plt.figure(figsize=(8, 6))
plt.scatter(cluster_sizes[:len(Pvddd)], Pvddd, s=10, alpha=0.7, color='purple', label="Power vs Cluster Size")
unique_cluster_sizes = np.arange(0, 14)  # Cluster sizes from 0 to 13
average_powers = []
valid_cluster_sizes = []
for size in unique_cluster_sizes:
    power_values = [Pvddd[i] for i in range(len(cluster_sizes)) if cluster_sizes[i] == size]
    if power_values: # Check if there are any power values for this cluster size
        valid_cluster_sizes.append(size)
        average_powers.append(np.mean(power_values))
# Plot the average power values as diamond markers
plt.scatter(valid_cluster_sizes, average_powers, color='red', marker='D', s=25, label="Average Power")
# Fit a straight line to the averaged points
slope, intercept, r_value, p_value, std_err = linregress(valid_cluster_sizes, average_powers)
fitted_line = slope * np.array(valid_cluster_sizes) + intercept
plt.plot(valid_cluster_sizes, fitted_line, color='blue', label=f"Fitted Line: y = {slope:.2f}x + {intercept:.2f}")
plt.title("Scatter Plot of Power vs Cluster Size")
plt.xlabel("Cluster Size (y-direction) [pixels]")
plt.ylabel("Power [$\mu$W]")
plt.grid()
plt.legend()
plt.savefig(f'{file_path}power_vs_clusterSize.png')

plt.figure(figsize=(8, 6))
plt.scatter(Vvddd, Pvddd, s=10, alpha=0.7, color='purple', label="Power vs Cluster Size")
plt.title("Scatter Plot of Power vs Vvddd")
plt.xlabel("Voltage (Vddd) [V]")
plt.ylabel("Power [$\mu$W]")
plt.grid()
plt.legend()
plt.savefig(f'{file_path}power_vs_Vvddd.png')

# debug double peak
if args.d is not None:
    cutoff = args.d
    event_peak1 = []
    event_peak2 = []

    for i in range(len(Pvddd)):
        if Pvddd[i] < cutoff:
            event_peak1.append(i)
        else:
            event_peak2.append(i)

    print("-----------------------------------")
    print(f"Events in peak 1 (<{cutoff} uW): {event_peak1}")
    print("-----------------------------------")
    print(f"Events in peak 2 (>={cutoff} uW): {event_peak2}")

    cluster_sizes_peak1 = [cluster_sizes[i] for i in event_peak1]
    cluster_sizes_peak2 = [cluster_sizes[i] for i in event_peak2]

    bins = range(0, 15)  # Covers all integer values from 0 to 14
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(cluster_sizes_peak1, bins=bins, color='blue', alpha=0.7, label="Peak 1", align='left')
    plt.title(f"Cluster Sizes for Events in Peak 1 (< {cutoff} uW)")
    plt.xlabel("Cluster Size (y-direction)")
    plt.ylabel("Counts")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(cluster_sizes_peak2, bins=bins, color='green', alpha=0.7, label="Peak 2", align='left')
    plt.title(f"Cluster Sizes for Events in Peak 2 (> {cutoff} uW)")
    plt.xlabel("Cluster Size (y-direction)")
    plt.ylabel("Counts")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{file_path}clusterSize_from_powerHistPeaks.png')
