import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

parser = argparse.ArgumentParser(description='Extract data and fit with Gaussian.')
parser.add_argument('-f', type=str, help='Path to the CSV file containing the data.')
args = parser.parse_args()

# Read the CSV file from the provided file path
csv_file = args.f
df = pd.read_csv(csv_file)

Ivddd_post = df["Ivddd_post"]
Ivddd_pre = df["Ivddd_pre"]
if "Noise" not in csv_file:
    Ivddd = (df["Ivddd_post"] - df["Ivddd_pre"]) * 100
    plot_xmin, plot_xmax = 0, 1300  # Set the x-axis range
    num_bins = 50
else:
    Ivddd = df["Ivddd_post"] * 100
    plot_xmin, plot_xmax = 3000, 4850  # Set the x-axis range
    num_bins = 175

Pvddd = Ivddd * 0.9 * 1000  # Calculate power in uW

# Define plot range

plt.xlim(plot_xmin, plot_xmax)

plt.hist(Pvddd, bins=num_bins, range=(plot_xmin, plot_xmax), density=True, alpha=0.6, color='g', label="Pvddd net histogram")

mean = np.mean(Pvddd)
rms = np.sqrt(np.mean(np.square(Pvddd - mean)))
fit_min = mean - rms * 3
fit_max = mean + rms * 3
Pvddd_filtered = Pvddd[(Pvddd >= fit_min) & (Pvddd <= fit_max)]
mu, std = norm.fit(Pvddd_filtered)

error_on_mean = std / np.sqrt(len(Pvddd_filtered))  # Error on fit mean
error_on_stat_mean = rms / np.sqrt(len(Pvddd))  # Error on statistical mean

x = np.linspace(plot_xmin, plot_xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f"Gaussian Fit\n$\mu={mu:.2e}, \sigma={std:.2e}$")

plt.title("Histogram of power values with Gaussian Fit")
plt.xlabel("Power [$\mu$W]")
plt.ylabel("Counts")
plt.grid()
plt.legend(loc='upper left')  # Position the legend in the top-left corner
plt.text(0.05, 0.80,  # Move text to the top-left corner
         f"Fit mean = {mu:.2e} ± {error_on_mean:.2e} $\mu$W\n"
         f"Stat. mean = {mean:.2e} ± {error_on_stat_mean:.2e} $\mu$W\n"
         f"Stat. std. dev. = {rms:.2e} $\mu$W", 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', horizontalalignment='left')
plt.show()
