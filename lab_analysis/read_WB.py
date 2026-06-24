import sys
import numpy as np
import pandas as pd
import argparse

args_parser = argparse.ArgumentParser(description='Read and process data arrays from CSV files.')
args_parser.add_argument('-a0', '--arr0', type=str, required=True, help='Path to the first data array CSV file.')
args_parser.add_argument('-a1', '--arr1', type=str, required=True, help='Path to the second data array CSV file.')
args_parser.add_argument('-wb', '--weightsbiases', default=None, type=str, help='Path to the original weights and biases file.')
args_parser.add_argument('-v', '--verbose', action='store_true', help='Print all parsed data from DATA arrays.')
args = args_parser.parse_args()

def readDataArray01(data_array0=None, data_array1=None):
    # Content of array 0 and 1
    test_structure = 24
    dnn_b5 = 12
    dnn_w5 = 696
    dnn_b2 = 232
    dnn_w2 = 3712
    pixel_config = 512
    arr_contents = [dnn_b5, dnn_w5, dnn_b2, dnn_w2, pixel_config]
    arr_contents_names = ["DNN Biases Layer 5", "DNN Weights Layer 5", "DNN Biases Layer 2", "DNN Weights Layer 2", "Pixel Config"]
    bit_length_conversion = 32/16 # during the read operations the int is converted to 32 bits, but each memory location is 16 bits

    # Load the saved csv files, convert to df, and delete the first column (as it is just the index)
    data_arr0 = pd.read_csv(data_array0, header=None, sep=',').iloc[:, 1:]
    data_arr1 = pd.read_csv(data_array1, header=None, sep=',').iloc[:, 1:]
    # Concat the two dataframes 
    data_combined = pd.concat([data_arr0, data_arr1], axis=0)
    # data_combined = data_combined.to_string(index=False, header=False)
    # data_combined = data_combined.iloc[:, ::-1]  # Invert the order of columns
    data_combined_np = data_combined.to_numpy() # Convert to numpy array
    data_combined_np = np.array([list(row[0]) for row in data_combined_np]) # Split the 32 numbers which is currently 'one' element, into separate elements
    data_combined_np = data_combined_np[:, ::-1]
    formatted_data = data_combined_np.flatten() # Flatten the 2D array to 1D array
    print("Total length of both data arrays = ", len(formatted_data))
    test_structure_arr = formatted_data[5164:5189]
    formatted_data = formatted_data[5188:] # Skip the first 5188 elements as it's junk/un-important data for this analysis
    if args.verbose:
        print("test structure: ", test_structure_arr)
        start_idx = 0
        for i, (content_length, content_name) in enumerate(zip(arr_contents, arr_contents_names)):
            arr_section = formatted_data[start_idx:start_idx+content_length]
            start_idx += content_length
            print(f"Array Section {i} ({content_name}):")
            print(arr_section)
            print("\n")
    
    # Compare with the original weights and biases file, if present.
    if args.weightsbiases is not None:
        original_wb = np.loadtxt(args.weightsbiases, delimiter=',')
        start_idx = 0
        total_parsed_length = sum(arr_contents)
        formatted_data_int = list(map(int, formatted_data)) # Convert all elements to int for comparison
        for i, (content_length, content_name) in enumerate(zip(arr_contents, arr_contents_names)):
            arr_section = formatted_data_int[start_idx:start_idx+content_length]
            original_section = original_wb[start_idx:start_idx+content_length]
            comparison = np.array_equal(arr_section, original_section)
            print(f"Comparison for {content_name}: {'Match' if comparison else 'Mismatch'}")
            start_idx += content_length
        assert len(original_wb) == total_parsed_length, f"Length of parsed data from DATA ARRAYS ({total_parsed_length}) are different from length of original weights and biases file ({len(original_wb)}). A change in expected word lengths is probably expected!"
    
data_array0 = args.arr0
data_array1 = args.arr1
readDataArray01(data_array0, data_array1)

