import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse DNN results from CSV files')
parser.add_argument("-r", "--readout", type=str, default='./readout.csv', help='Path to readout CSV file')
parser.add_argument("-d", "--dnn_rtl", type=str, default='./dnn_RTL_out.csv', help='Path to DNN RTL output CSV file')
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
