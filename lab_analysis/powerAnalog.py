import numpy as np
import matplotlib.pyplot as plt
import sys
csv_path = "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2026.02.26_17.30.17_AnalogPower/analogPower_avg.csv"
def plot_analog_power(csv_path):
    # Load the data, skipping the header
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    # Ibias is always the first column
    Ibias = data[:, 0]
    current = data[:, 1]

    # Find unique Ibias values
    unique_Ibias = np.unique(Ibias)

    # Initialize lists to store results
    avg_current = []
    std_dev_current = []

    # Calculate average and standard deviation for each Ibias
    for ib in unique_Ibias:
        # Filter current values corresponding to the current Ibias
        current_values = current[Ibias == ib]
        avg_current.append(np.mean(current_values))
        std_dev_current.append(np.std(current_values))

    # Convert results to numpy arrays for plotting
    avg_current = np.array(avg_current)
    std_dev_current = np.array(std_dev_current)

    # Plot each Ivdda iteration
    plt.figure(figsize=(8, 6))
    plt.errorbar(unique_Ibias, avg_current, yerr=std_dev_current, fmt='o', capsize=5)
    plt.title("Analog Power Sweep: Ivdda vs Ibias")
    plt.xlabel("Ibias (V)")
    plt.ylabel("Ivdda (A)")
    plt.grid(True)
    # plt.legend(title="Iteration")
    plt.tight_layout()

    # Save the figure
    plt.savefig("analog_power.png", dpi=300)

    plt.show()

plot_analog_power(csv_path)






