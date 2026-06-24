import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress
from scipy.optimize import curve_fit
import argparse
import sys, os
from SmartPixStyle import *
import mplhep as hep
from Analyze import *
import json
import glob
hep.style.use("ATLAS")


parser = argparse.ArgumentParser(description='Extract data and fit with Gaussian.')
parser.add_argument('-f', type=str, help='Path to the CSV file containing the power measurement data.')
parser.add_argument('-c', '--compout', type=str, help='Path to the compout CSV file.', default='/asic/projects/C/CMS_PIX_28/dshekar/filter/model_pipeline/tmp_16x16_centeredIncidence/1000_1600_2400/compouts_ylocal_0.00_1.35.csv')
parser.add_argument('-r', type=int, default=1000, help='Maximum range of power values in plot.')
parser.add_argument('-d', type=int, default=None, help='Debug double peak if passed. Value passed as argument will be chosen as the cut between the two peaks.')
parser.add_argument('-p', action='store_true', default=False, help='Path to load JSON files and make plot of power vs noise-threshold.')
args = parser.parse_args()

def get_y_cluster_size(event_matrix):
    # event_matrix: 2D numpy array of shape (16, 16)
    # Returns number of rows containing at least one nonzero
    return np.count_nonzero(event_matrix.sum(axis=1))

def parse_data(file_path, xlimit):
    # Read the CSV file from the provided file path
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
        plot_xmin, plot_xmax = 0, xlimit  # Set the x-axis range
        num_bins = 50
    else:
        Ivddd = df["Ivddd_post"] * 100
        plot_xmin, plot_xmax = 3000, 4850  # Set the x-axis range
        num_bins = 175

    return Ivddd_pre, Ivddd, plot_xmin, plot_xmax, num_bins, event_number, Vvddd

def plot_figure(X, Y, plot_type, label, color, marker, xlabel, ylabel, title, file_name, xlimLow = None, xlimHigh = None):
    plt.figure(figsize=(8, 6))
    for X_iter, Y_iter, plot_type_iter, label_iter, color_iter, marker_iter in zip(X, Y, plot_type, label, color, marker):
        if plot_type_iter == 'line':
            plt.plot(X_iter, Y_iter, label=label_iter, color=color_iter)
        elif plot_type_iter == 'scatter':
            plt.scatter(X_iter, Y_iter, label=label_iter, color=color_iter, marker=marker_iter, s=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.savefig(file_name)


if args.p:
    # Simple 3‑parameter exponential: y = A * exp(B x) + C
    def exp3(x, a, b, c):
        # a * exp(b x) + c
        return a * np.exp(b * x) + c

    # HARD CODED json file path for performance vs noise threshold
    preformance_json = '/asic/projects/C/CMS_PIX_28/dshekar/filter/model_pipeline/performance_vs_noise-threshold.json'
    
    # Extracting and plotting results of DNN power vs noise threshold
    noise_thresholds = [300, 400, 500, 600, 700, 800, 900, 1000]
    # HARD CODED file paths for each noise threshold based on runs from January 2026
    file_mother_paths = ['/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/Phit_'+str(threshold)+'e-VTH_synchronizedInjection/' for threshold in noise_thresholds]
    dnn_power_dirs = [
    dir_path
    for path in file_mother_paths
    for dir_path in glob.glob(path + 'ChipVersion1_ChipID23_SuperPix1/*_dnnPowerOnHit/')
    ]
    power_mean = []
    power_sigma = []
    thresholds = []
    for dir_path in dnn_power_dirs:
        json_files = glob.glob(os.path.join(dir_path, 'power_results.json'))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                power_mean.append(data['statistical_mean']['value'])
                power_sigma.append(data['statistical_mean']['error'])#(data['statistical_stddev'])#
                thresholds.append(data['noise_threshold'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.errorbar(thresholds, power_mean, yerr=power_sigma, fmt='o', color='black', capsize=5, markerfacecolor='red', label='Power measurement')
    ax1.plot(thresholds, power_mean, color='red')
    thresholds = np.array(thresholds)
    power_mean = np.array(power_mean)
    power_sigma = np.array(power_sigma)
    mask = thresholds >= 500
    x = np.array(thresholds[mask], dtype=float) / 1000.0  # 0.5–1.0
    y = np.array(power_mean[mask], dtype=float)
    yerr = np.array(power_sigma[mask], dtype=float)

    # Convert lists to arrays
    thresholds = np.array(thresholds, dtype=float)
    power_mean = np.array(power_mean, dtype=float)
    power_sigma = np.array(power_sigma, dtype=float)

    # Use only thresholds >= 500 e-
    mask = thresholds >= 500
    x = thresholds[mask] / 1000.0
    y = power_mean[mask]
    yerr = power_sigma[mask]

    # Initial guesses: a ~ dynamic range, b ~ small curvature, c ~ min value
    a0 = y.max() - y.min()
    b0 = 0.0
    c0 = y.min()
    p0 = [a0, b0, c0]
    # Optional bounds to keep it well-behaved
    bounds = ([0.0, -10.0, 0.0], [1e6, 10.0, 1e6])
    popt, pcov = curve_fit( exp3, x, y, p0=p0, sigma=yerr, absolute_sigma=True, bounds=bounds, maxfev=10000)
    a, b, c = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    # Make smooth curve in original threshold units
    x_fit_full = np.linspace(300, 1100, 400)
    x_for_model = x_fit_full / 1000.0   # only if fit used thresholds/1000.0
    y_fit_full = exp3(x_for_model, a, b, c)

    #ax1.plot(x_fit_full, y_fit_full, 'r-', label=(r'Power measurement fit: $a e^{b x/1000} + c$' '\n' rf'$a$={a:.2f}±{a_err:.2f}, ' rf'$b$={b:.2f}±{b_err:.2f}, ' rf'$c$={c:.2f}±{c_err:.2f}'))
    ax1.set_xlabel("Noise threshold [e-]", fontsize=14)
    ax1.set_ylabel("Power [$\mu$W]", color='r', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlim(min(noise_thresholds) - 100 , max(noise_thresholds) + 200)
    ax1.set_ylim(0, 900)  # Set y-axis limits
    ax1.tick_params(axis='x', pad=10)  # Add space between x-axis labels and plot boundary
    ax1.grid()

    # Extracting and plotting results of DNN performance vs noise threshold
    with open(preformance_json, 'r') as f:
        data = json.load(f)
    # Order keys based on noise threshold
    ordered_keys = sorted(data.keys(), key=lambda k: float(k.split(",")[0]))
    ordered_data = {key: data[key] for key in ordered_keys}
    
    noise_thresholds = []
    signal_efficiencies = []
    background_rejections = []
    data_reductions = []
    rtl_matches = []
    for key, value in ordered_data.items():
        thresholds = list(map(float, key.split(",")))
        noise_thresholds.append(thresholds[0]) # Load noise threshold
        signal_efficiencies.append(value["signal_efficiency"]/100)
        background_rejections.append(value["background_rejection"]/100)
        data_reductions.append(value["data_reduction"]/100)
        rtl_matches.append(value["dnn_rtl_match_ts19"]/100)

    ax2 = ax1.twinx()  
    ax2.plot(noise_thresholds, signal_efficiencies, 's', color='black', label='Signal efficiency', markerfacecolor='blue')
    ax2.plot(noise_thresholds, data_reductions, 'd', color='black', label='Data reduction', markerfacecolor='green')
    ax2.plot(noise_thresholds, signal_efficiencies, color='blue')
    ax2.plot(noise_thresholds, data_reductions, color='green')
    # Fit signal efficiency vs noise threshold with inverted exponential
    noise_thresholds = np.array(noise_thresholds, dtype=float)
    signal_efficiencies = np.array(signal_efficiencies, dtype=float)
    data_reductions = np.array(data_reductions, dtype=float)

    x = noise_thresholds/1000
    y = signal_efficiencies
    def fit_func(x, a, b, c):
        return a - b * (np.exp(-c * x))
    # Fit the model to your data
    p0 = [1, 1, 1] 
    bounds = ([0,0,0], [1000, 1000, 1000])
    popt, pcov = curve_fit(fit_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
    a_fit, b_fit, c_fit = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    x_fit_full = np.linspace(300, 1100, 400)
    x_for_model = x_fit_full / 1000.0   # only if fit used thresholds/1000.0
    y_fit_full = fit_func(x_for_model, a_fit, b_fit, c_fit)
    #ax2.plot(x_fit_full, y_fit_full, 'b-', label=(r'Signal efficiency fit: $a - be^{-c x/1000}$' '\n' rf'$a$={a_fit:.2f}±{a_err:.2f}, ' rf'$b$={b_fit:.2f}±{b_err:.2f}, ' rf'$c$={c_fit:.2f}±{c_err:.2f}'))
    
    x = noise_thresholds/1000
    y = data_reductions
    # Initial guesses: a ~ dynamic range, b ~ small curvature, c ~ min value
    a0 = y.max() - y.min()
    b0 = 0.0
    c0 = y.min()
    p0 = [a0, b0, c0]
    # Optional bounds to keep it well-behaved
    bounds = ([0.0, -10.0, 0.0], [1e6, 10.0, 1e6])
    popt, pcov = curve_fit( exp3, x, y, p0=p0, bounds=bounds, maxfev=10000)
    a, b, c = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    # Make smooth curve in original threshold units
    x_fit_full = np.linspace(300, 1100, 400)
    x_for_model = x_fit_full / 1000.0   # only if fit used thresholds/1000.0
    y_fit_full = exp3(x_for_model, a, b, c)
    #ax2.plot(x_fit_full, y_fit_full, 'b--', label=(r'Data reduction fit: $a e^{b x/1000} + c$' '\n' rf'$a$={a:.2f}±{a_err:.2f}, ' rf'$b$={b:.2f}±{b_err:.2f}, ' rf'$c$={c:.2f}±{c_err:.2f}'))

    # ax2.set_ylabel("Neural network performance", color='blue', fontsize=14)
    # ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel("")  # Remove default label
    ax2.text(1.07, 0.85, "Signal efficiency", color='blue', fontsize=14, transform=ax2.transAxes, va='center', ha='left', rotation=90)
    ax2.text(1.07, 0.65, "and", color='black', fontsize=14, transform=ax2.transAxes, va='center', ha='left', rotation=90)
    ax2.text(1.07, 0.46, "Data reduction", color='green', fontsize=14, transform=ax2.transAxes, va='center', ha='left', rotation=90)
    ax2.set_ylim(0, 1)  # Set y-axis limits

    # Update legend with some slight re-ordering to include both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Desired order: Power measurement, Power fit, Signal efficiency, Signal efficiency fit
    all_lines = [lines[0], lines2[0], lines2[1]]
    # all_lines = [lines[1], lines[0], lines2[0], lines2[2], lines2[1], lines2[3]]
    all_labels = [labels[0], labels2[0], labels2[1]]
    # all_labels = [labels[1], labels[0], labels2[0], labels2[2], labels2[1], labels2[3]]
    legend = ax1.legend(all_lines, all_labels, loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=True, labelspacing=1.2)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)

    fig.tight_layout()
    SmartPixLabel(ax1, 0, 1.003, size=18)
    # plt.title("DNN characteristics vs noise threshold")
    plt.savefig(f'./dnnResults_vs_noise_threshold.png')#f'{file_path}dnnResults_vs_noise_threshold.png')

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(noise_thresholds, rtl_matches, 's', color='black', markerfacecolor='black')
    ax1.set_xlabel("Noise threshold [e-]", fontsize=14)
    ax1.set_ylabel("Fraction of matching results (on-chip NN/RTL)", fontsize=14)
    # ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlim(min(noise_thresholds) - 100 , max(noise_thresholds) + 200)
    # ax1.set_ylim(0, 900)  # Set y-axis limits
    ax1.tick_params(axis='x', pad=10)  # Add space between x-axis labels and plot boundary
    ax1.grid()
    fig.tight_layout()
    SmartPixLabel(ax1, 0, 1.003, size=18)
    # plt.title("DNN characteristics vs noise threshold")
    plt.savefig(f'./dnnRTLmatch_vs_noise_threshold.png')#f'{file_path}dnnResults_vs_noise_threshold.png')
    sys.exit(0)

file_path = os.path.join(args.f, '')
xlimit = args.r
Ivddd_pre, Ivddd, plot_xmin, plot_xmax, num_bins, event_number, Vvddd = parse_data(file_path, xlimit)

plot_figure(X=[range(len(Ivddd_pre))], Y=[Ivddd_pre*100], plot_type = ['scatter'], label=['Ivddd_pre'], color=['blue'], marker=['o'], xlabel='Event number', ylabel='Ivddd_pre [A]', title='Scatterplot of Ivddd_pre', file_name=f'{file_path}IvdddPre_vs_event.png')

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
# Create a dictionary to hold the statistics
stats = {
    "num_events": len(Pvddd),
    "noise_threshold": int(file_path.split('Phit_')[1].split('e-')[0]),
    "statistical_mean": {
        "value": mean,
        "error": error_on_stat_mean
    },
    "statistical_stddev": rms,
    "fit_mean": {
        "value": mu,
        "error": error_on_mean
    },
    "fit_stddev": std
}

# Write the statistics to a JSON file
json_file_path = os.path.join(file_path, 'power_results.json')
with open(json_file_path, 'w') as json_file:
    json.dump(stats, json_file, indent=4)
print(f"Statistics saved to {json_file_path}")

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
unique_cluster_sizes = np.arange(0, 14)  # Cluster sizes from 0 to 13
average_powers = []
valid_cluster_sizes = []
for size in unique_cluster_sizes:
    power_values = [Pvddd[i] for i in range(len(cluster_sizes)) if cluster_sizes[i] == size]
    if power_values: # Check if there are any power values for this cluster size
        valid_cluster_sizes.append(size)
        average_powers.append(np.mean(power_values))
# Fit a straight line to the averaged points
slope, intercept, r_value, p_value, std_err = linregress(valid_cluster_sizes, average_powers)
fitted_line = slope * np.array(valid_cluster_sizes) + intercept
plot_figure(X=[cluster_sizes[:len(Pvddd)], valid_cluster_sizes, valid_cluster_sizes], Y=[Pvddd, average_powers, fitted_line], plot_type=['scatter', 'scatter', 'line'], label=['Power vs Cluster Size', 'Average Power', f"Fitted Line: y = {slope:.2f}x + {intercept:.2f}"], color=['purple', 'darkorange', 'blue'], marker=['o', 'd', None], xlabel="Cluster Size (y-direction) [pixels]", ylabel="Power [$\mu$W]", title="Scatter Plot of Power vs Cluster Size", file_name=f'{file_path}power_vs_clusterSize.png')

plot_figure(X=[Vvddd], Y=[Pvddd], plot_type = ['scatter'], label=['Power vs Cluster Size'], color=['purple'], marker=['o'], xlabel="Voltage (Vddd) [V]", ylabel="Power [$\mu$W]", title="Scatter Plot of Power vs Vvddd", file_name=f'{file_path}power_vs_Vvddd.png')

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
