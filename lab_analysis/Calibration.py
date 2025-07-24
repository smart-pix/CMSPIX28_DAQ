import numpy as np
arrays = []

for i in range(256):
    pixel = np.load(f'/asic/projects/C/CMS_PIX_28/benjamin/testing/workarea_112024/CMSPIX28_DAQ/spacely/PySpacely/calibrationNPix{i}.npy')
    arrays.append(pixel)

matrix = np.hstack(arrays)
np.save('/asic/projects/C/CMS_PIX_28/benjamin/testing/workarea_112024/CMSPIX28_DAQ/spacely/PySpacely/calibrationNPix.npy', matrix)

pattern = np.tile([1, 1, 0], 256)  # Create the repeating pattern: [1, 1, 0] * 256
#matches = np.all(matrix == pattern, axis=1)

# Step 1: Check for an exact match where the 'X' can be either 0 or 1
matches = np.all((matrix[:, ::3] == 1) & (matrix[:, 1::3] == 1), axis=1)  # Check every 1st and 2nd bit for 1,1

# Since 'X' can be either 0 or 1, allow it in the 3rd column of each triplet
# Step 1: Check if any row in the array matches the pattern (ignoring the third bit in every triplet)
matches_with_x = np.all(matrix[:, ::3] == 1, axis=1) & np.all(matrix[:, 1::3] == 1, axis=1)  # Check every 1st and 2nd bit for 1,1

# If there is an exact match, print the indices
if np.any(matches_with_x):
    print("Exact match found at indices:", np.where(matches_with_x)[0])
else:
    print("No exact match found.")

    # Step 2: If no exact match, find the closest match using Euclidean distance
    # Calculate the Euclidean distance between the pattern and each row in the array
    distances = np.linalg.norm(matrix - pattern, axis=1)

    BitError = distances**2

    print(f"there are at best {BitError} bit different from targer")
    


    # Find the index of the row with the smallest distance
    closest_index = np.argmin(distances)

    print(f"Closest match found at index: {closest_index}")
    print(f"Distance: {distances[closest_index]}")

    # Step 3: Report the number of errors and column indices that do not match
    closest_row = matrix[closest_index]

    # Find where the closest row doesn't match the pattern
    incorrect_columns = np.where(closest_row != pattern)[0]
    
    print(f"Number of errors (bits differing): {len(incorrect_columns)}")
    print(f"Column indices that do not match: {incorrect_columns}")