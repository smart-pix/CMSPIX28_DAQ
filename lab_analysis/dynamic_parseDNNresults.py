import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse DNN results from CSV files')
parser.add_argument("-r", "--readout", type=str, default='./readout.csv', help='Path to readout CSV file')
parser.add_argument("-d", "--dnn_rtl", type=str, default='./dnn_RTL_out.csv', help='Path to DNN RTL output CSV file')
parser.add_argument("-p", "--true_pt", type=str, default=None, help='Path to true pt values for inputs')
args = parser.parse_args()

dnn0 = [0,0]
dnn1 = [0,1]
dnn2 = [1,0]
possible_dnn_outputs = [dnn0, dnn1, dnn2]
shape_per_event_results = 4,64
len_dnn_data = 40
len_empty_bits = 24

def eval_dnn_result(results_file, bit_iter):
    final_results = []
    for evt_iter in range(len(results_file)):
        event_output = results_file[evt_iter]
        reshaped_output = event_output.reshape(shape_per_event_results)
        dnn0_clipped = reshaped_output[-1][len_empty_bits:]
        dnn1_clipped = reshaped_output[-2][len_empty_bits:]
        lsb = dnn0_clipped[bit_iter]
        msb = dnn1_clipped[bit_iter]
        event_result = [msb, lsb]
        if event_result not in possible_dnn_outputs:
            raise ValueError(f"Event result {event_result} does not match with possible values {possible_dnn_outputs}")
        else:
            matching_index = possible_dnn_outputs.index(event_result)
            final_results.append(matching_index)

    final_results = np.array(final_results)
    return final_results

results_file_to_evaluate = np.genfromtxt(args.readout, delimiter=',', dtype=int)
results_file = results_file_to_evaluate[:200]

dnn_RTL_out = np.genfromtxt(args.dnn_rtl, delimiter=',', dtype=int)
expected_train_results = dnn_RTL_out[:200]

assert len(results_file) == len(expected_train_results), "Mismatch in number of events between results file and expected train results"
assert len(results_file_to_evaluate) == len(dnn_RTL_out), "Mismatch in number of events between results file to evaluate and DNN RTL output"
assert len(results_file) <= len(results_file_to_evaluate), "Evaluation set should have more events than the train set"
assert len(expected_train_results) <= len(dnn_RTL_out), "Evaluation set should have more events than the train set"

train_result_vs_time = np.zeros((len_dnn_data, len(results_file)), dtype=int)

for evt_iter in range(len(results_file)):
    event_output = results_file[evt_iter]
    reshaped_output = event_output.reshape(shape_per_event_results)
    dnn0_clipped = reshaped_output[-1][len_empty_bits:]
    dnn1_clipped = reshaped_output[-2][len_empty_bits:]
    for i in range(len_dnn_data):
        lsb = dnn0_clipped[i]
        msb = dnn1_clipped[i]
        event_result = [msb, lsb]
        if event_result not in possible_dnn_outputs:
            matching_index = 999
        else:
            matching_index = possible_dnn_outputs.index(event_result)
        train_result_vs_time[i, evt_iter] = matching_index

assert train_result_vs_time.shape == (len_dnn_data, len(results_file)), "Shape mismatch in train_result_vs_time"

score = []

for i in range(len_dnn_data):
    score.append(np.sum(np.abs(train_result_vs_time[i] - expected_train_results)))

score = np.array(score)
print(score)
print("Best score = ", np.min(score), ", and time stamp of best score = ", np.argmin(score))
print("Passing time stamp = ", np.argmin(score) + 1, "(evaluating at next time stamp to ensure evaluation in a safe output time range of dnn0 and dnn1).")
final_results = eval_dnn_result(results_file_to_evaluate, np.argmin(score)+1)
np.savetxt('final_results.csv', final_results, delimiter=',', fmt='%d')

assert final_results.shape == dnn_RTL_out.shape

no_match = np.where(final_results != dnn_RTL_out)
for i in no_match:
    print("Final result:", final_results[i])
    print("DNN RTL out:", dnn_RTL_out[i])

matches = np.sum(final_results == dnn_RTL_out)
print(final_results.shape, matches)
total = len(final_results)
percentage_match = (matches / total) * 100

print(f"Percentage of matches: {percentage_match:.2f}%")

# load the true pt values if provided
if args.true_pt:

    # remove last two extra entries
    final_results = final_results[:-2]
    dnn_RTL_out = dnn_RTL_out[:-2]

    # get the test vectors with exact y-profile matches
    import sys
    sys.path.append("../spacely/PySpacely/spacely-asic-config/CMSPIX28Spacely")
    from CMSPIX28Spacely_Subroutines_A0 import input_bin_to_y_profile
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("ATLAS")
    from SmartPixStyle import *
    from Analyze import inspectPath
    import os
    from matplotlib.ticker import MultipleLocator

    info = inspectPath(os.path.dirname(args.readout))
    print(info)

    # load true y-profile
    input_bin="/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_A/tb/dnn/csv/l6/input_bin.csv"
    yprofile = input_bin_to_y_profile(input_bin)
    print("input_bin:", input_bin, yprofile.shape)

    # load y-profile from ASIC
    asic = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID17_SuperPix1/2025.06.15_08.37.02_DNN/yprofiles.csv"
    yprofile_asic = np.loadtxt(asic, delimiter=",").astype(int)
    print("asic yprofile:", asic, yprofile_asic.shape)
    yprofile_asic = yprofile_asic[:-2] # last two are not real, just padded during testing
    print("asic yprofile removing last 2 padded for testing:", asic, yprofile_asic.shape)

    # compare them
    eq = (yprofile_asic == yprofile)
    alltrue = np.where(np.all(eq,axis=1))[0] # find where all values of y-profile are equal
    print("Test vectors that have exactly equal yprofiles:", alltrue.shape, alltrue)
    print("Examples below ...")
    for iT, i in enumerate(alltrue):
        if iT>10: break
        print("input_bin:", yprofile[i])
        print("     asic:", yprofile_asic[i])
        print()

    # print fraction matching
    print("Fraction exactly matching:", alltrue.shape[0]/yprofile.shape[0])

    # get the true pt values
    true_pt = np.genfromtxt(args.true_pt, delimiter=',', skip_header=1)
    print(f"Loaded {len(true_pt)} true pt values")
    
    # take the first N entries to match final_results shape
    true_pt = true_pt[:len(final_results)]
    print(true_pt.shape)

    # Load the reference results for comparison
    ref_results_file = "/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_A/tb/dnn/csv/l6/layer7_out_ref_int.csv"
    ref_results = np.genfromtxt(ref_results_file, delimiter=',', dtype=int)
    print(f"Loaded {len(ref_results)} reference results from {ref_results_file}", "shape:", ref_results.shape)

    # only consider the test vectors inside of alltrue
    print("Filtering true_pt and final_results to only include matching test vectors ...")
    print(true_pt.shape, final_results.shape)
    
    # Work with extracted subset (matching test vectors)
    true_pt_extracted = true_pt[alltrue]
    final_results_extracted = final_results[alltrue]
    print(final_results_extracted)
    dnn_RTL_out_extracted = dnn_RTL_out[alltrue]
    ref_results_extracted = ref_results[alltrue]
    print("Extracted shapes:", true_pt_extracted.shape, final_results_extracted.shape)

    # Calculate fraction of correctly labeled high pT particles vs true pT
    # Create bins for true pT values from -5 to 5
    pt_bins = np.linspace(-5, 5, 101)
    bin_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    
    # For extracted subset
    asic_fractions_extracted = []
    offline_fractions_extracted = []
    ref_results_extracted_fractions = []
    
    # For full lists
    asic_fractions_full = []
    offline_fractions_full = []
    ref_results_full_fractions = []

    for i in range(len(pt_bins) - 1):
        # EXTRACTED SUBSET
        # Find particles in this pT bin with |pT| > 0.2 GeV
        in_bin_extracted = (true_pt_extracted >= pt_bins[i]) & (true_pt_extracted < pt_bins[i+1])
        print(f"Extracted bin [{pt_bins[i]:.2f}, {pt_bins[i+1]:.2f}): {np.sum(in_bin_extracted)} particles")

        if np.sum(in_bin_extracted) > 0:
            # Check how many were correctly labeled as high pT (label 0)
            labeled_high_pt = (final_results_extracted[in_bin_extracted] == 0)
            asic_fraction_correct = np.sum(labeled_high_pt) / np.sum(in_bin_extracted)
            asic_fractions_extracted.append(asic_fraction_correct)

            # do the same for offline results
            offline_labeled_high_pt = (dnn_RTL_out_extracted[in_bin_extracted] == 0)
            offline_fraction_correct = np.sum(offline_labeled_high_pt) / np.sum(in_bin_extracted)
            offline_fractions_extracted.append(offline_fraction_correct)

            # do the same for reference results
            ref_labeled_high_pt = (ref_results_extracted[in_bin_extracted] == 0)
            ref_fraction_correct = np.sum(ref_labeled_high_pt) / np.sum(in_bin_extracted)
            ref_results_extracted_fractions.append(ref_fraction_correct)
        else:
            asic_fractions_extracted.append(0)
            offline_fractions_extracted.append(0)
            ref_results_extracted_fractions.append(0)

        # FULL LISTS
        # Find particles in this pT bin with |pT| > 0.2 GeV
        in_bin_full = (true_pt >= pt_bins[i]) & (true_pt < pt_bins[i+1])
        print(f"Full bin [{pt_bins[i]:.2f}, {pt_bins[i+1]:.2f}): {np.sum(in_bin_full)} particles")

        if np.sum(in_bin_full) > 0:
            # Check how many were correctly labeled as high pT (label 0)
            labeled_high_pt_full = (final_results[in_bin_full] == 0)
            asic_fraction_correct_full = np.sum(labeled_high_pt_full) / np.sum(in_bin_full)
            asic_fractions_full.append(asic_fraction_correct_full)

            # do the same for offline results
            offline_labeled_high_pt_full = (dnn_RTL_out[in_bin_full] == 0)
            offline_fraction_correct_full = np.sum(offline_labeled_high_pt_full) / np.sum(in_bin_full)
            offline_fractions_full.append(offline_fraction_correct_full)

            # do the same for reference results
            print(ref_results.shape, in_bin_full.shape)
            ref_labeled_high_pt_full = (ref_results[in_bin_full] == 0)
            ref_fraction_correct_full = np.sum(ref_labeled_high_pt_full) / np.sum(in_bin_full)
            ref_results_full_fractions.append(ref_fraction_correct_full)
        else:
            asic_fractions_full.append(0)
            offline_fractions_full.append(0)
            ref_results_full_fractions.append(0)

    # Use extracted subset for plotting (as in original code)
    asic_fractions = asic_fractions_extracted
    offline_fractions = offline_fractions_extracted

        # if fractions[-1] == 0:
        #     fractions[-1] = None

    # Create the plot
    fig, ax = plt.subplots(figsize=(6,6))
    print(bin_centers)
    print(asic_fractions)
    print(offline_fractions)
    ax.plot(bin_centers, ref_results_full_fractions, 'd', linewidth=1, markersize=4, color="black", label="Layer7 Ref Full Stat.")
    # ax.plot(bin_centers, ref_results_extracted_fractions, 'D', linewidth=1, markersize=4, color="green", label="Layer7 Ref")
    # ax.plot(bin_centers, asic_fractions, '-o', linewidth=1, markersize=4, color="red", label="ROIC")
    # ax.plot(bin_centers, offline_fractions, '1', linewidth=1, markersize=6, color="blue", label="DNN RTL")    
    # ax.stairs(fractions, pt_bins, linewidth=2, color="black", alpha=0.8)
    ax.set_xlabel(r'True $p_{\mathrm{T}}$ [GeV]')
    ax.set_ylabel(r'Predicts high $p_{\mathrm{T}}$ (0) / particles in bin')
    # ax.set_title('DNN Performance: Fraction of correctly labeled high pT particles vs true pT')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlim(-5, 5)
    ax.tick_params(which='minor', length=4)
    ax.tick_params(which='major', length=6)
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    # Add text annotation with matching statistics
    SmartPixLabel(ax, 0.05, 0.9, size=18)
    ax.text(0.05, 0.86, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=12, color="black", ha='left', va='bottom')
    ax.text(0.05, 0.82, f'y-profile matches: {alltrue.shape[0]}/{yprofile.shape[0]} ({alltrue.shape[0]/yprofile.shape[0]:.3f})', transform=ax.transAxes, fontsize=10, ha='left', va='bottom')
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('high_pt_fraction_vs_true_pt.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
                                    

