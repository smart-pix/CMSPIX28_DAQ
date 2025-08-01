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

    # function to perform analysis of efficiency results
    def analyzeResults(pt, pt_bins, results):
        effs, eff_errs = [], []
        # loop over the pT bins
        for i in range(len(pt_bins) - 1):
            in_bin_extracted = (pt >= pt_bins[i]) & (pt < pt_bins[i+1])
            N = np.sum(in_bin_extracted)
            if np.sum(in_bin_extracted) > 0:
                # Check how many were correctly labeled as high pT (label 0)
                labeled_high_pt = (results[in_bin_extracted] == 0)
                npass = np.sum(labeled_high_pt)
                eff = npass / N
                eff_err = efficiency_error(eff, N)
                # save
                effs.append(eff)
                eff_errs.append(eff_err)
            else:
                effs.append(0)
                eff_errs.append(0)
        return np.array(effs), np.array(eff_errs)
        
    # efficiency error calculation
    def efficiency_error(eff, N):
        ''' 
        See section 2.2.1 https://lss.fnal.gov/archive/test-tm/2000/fermilab-tm-2286-cd.pdf
        eff = estimate of the efficiency
        N = sample size
        '''
        return np.sqrt(eff * (1 - eff) / N)

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

    # Create bins for true pT values from -4 to 4
    bins_pos = np.unique(np.concatenate([
        np.linspace(0, 1, 5, endpoint=False),
        np.linspace(1, 2, 5, endpoint=False),
        np.linspace(2, 4, 3, endpoint=True)
    ]))
    bins_neg = -1 * bins_pos[::-1]
    pt_bins = np.unique(np.concatenate([bins_neg, bins_pos]))
    bin_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    bin_widths = np.diff(pt_bins)

    # asic results
    asic_fractions_full, asic_errors_full = analyzeResults(true_pt, pt_bins, final_results)
    asic_fractions_extracted, asic_errors_extracted = analyzeResults(true_pt[alltrue], pt_bins, final_results[alltrue])

    # offline results
    offline_fractions_full, offline_errors_full = analyzeResults(true_pt, pt_bins, dnn_RTL_out)
    offline_fractions_extracted, offline_errors_extracted = analyzeResults(true_pt[alltrue], pt_bins, dnn_RTL_out[alltrue])

    # reference results
    ref_fractions_full, ref_errors_full = analyzeResults(true_pt, pt_bins, ref_results)
    ref_fractions_extracted, ref_errors_extracted = analyzeResults(true_pt[alltrue], pt_bins, ref_results[alltrue])

    # Create the plot
    fig, ax = plt.subplots(figsize=(6,6))
    
    # plot error bars
    ax.errorbar(bin_centers, ref_fractions_full, xerr=bin_widths/2, yerr=asic_errors_full, fmt='o', linewidth=1, markersize=4, color="black", label=f"Layer7 Ref Full Stat.") #Full Stat.") # "d"
    ax.errorbar(bin_centers, ref_fractions_extracted, xerr=bin_widths/2, yerr=ref_errors_full, fmt='o', linewidth=1, markersize=4, color="green", label="Layer7 Ref") # "D"
    ax.errorbar(bin_centers, asic_fractions_extracted, xerr=bin_widths/2, yerr=asic_errors_extracted, fmt='o', linewidth=1, markersize=4, color="red", label=f"ROIC ({alltrue.shape[0]}/{yprofile.shape[0]})", alpha=0.5) # f"ROIC ({alltrue.shape[0]}/{yprofile.shape[0]})"
    ax.errorbar(bin_centers, offline_fractions_extracted, xerr=bin_widths/2, yerr=offline_errors_extracted, fmt='1', linewidth=1, markersize=6, color="blue", label="DNN RTL")    
    
    # set plot styles
    ax.set_xlabel(r'True $p_{\mathrm{T}}$ [GeV]')
    # ax.set_ylabel(r'Predicts high $p_{\mathrm{T}}$ (0) / particles in bin')
    ax.set_ylabel(r'Fraction predicted high $p_{\mathrm{T}}$ (> 0.2 GeV)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-4, 4)
    ax.tick_params(which='minor', length=4)
    ax.tick_params(which='major', length=6)
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    # Add text annotation with matching statistics
    # SmartPixLabel(ax, 0.05, 0.9, size=14)
    # ax.text(0.05, 0.86, f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", transform=ax.transAxes, fontsize=10, color="black", ha='left', va='bottom')
    SmartPixLabel(ax, 0, 1.0, text=f"ROIC V{int(info['ChipVersion'])}, ID {int(info['ChipID'])}, SuperPixel {int(info['SuperPix'])}", size=12, fontweight='normal', style='normal')
    # ax.text(0.05, 0.82, f'y-profile matches: {alltrue.shape[0]}/{yprofile.shape[0]} ({alltrue.shape[0]/yprofile.shape[0]:.3f})', transform=ax.transAxes, fontsize=10, ha='left', va='bottom')
    ax.legend(loc='lower left', fontsize=11, bbox_to_anchor=(0, 0))
    plt.tight_layout()
    plt.savefig('high_pt_fraction_vs_true_pt.pdf', dpi=300, bbox_inches='tight')
                                    

