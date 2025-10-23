import numpy as np
import matplotlib.pyplot as plt
import sys
csv_path = "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.10.23_12.07.31_AnalogPower/analogPower.csv"
def plot_analog_power(csv_path):
    # Load the data, skipping the header
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    # Ibias is always the first column
    

    # Plot each Ivdda iteration
    plt.figure(figsize=(8, 6))
    for col in col_names[1:]:
        plt.plot(ibias, data[col], label=col)

    plt.title("Analog Power Sweep: Ivdda vs Ibias")
    plt.xlabel("Ibias (V)")
    plt.ylabel("Ivdda (A)")
    plt.grid(True)
    plt.legend(title="Iteration")
    plt.tight_layout()
    plt.show()

plot_analog_power(csv_path)