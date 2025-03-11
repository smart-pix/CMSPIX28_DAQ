import numpy as np
arrays = []

for i in range(256):
    pixel = np.load(f'/asic/projects/C/CMS_PIX_28/benjamin/testing/workarea_112024/CMSPIX28_DAQ/spacely/PySpacely/calibrationNPix{i}.npy')
    arrays.append(pixel)

matrix = np.hstack(arrays)
np.save('/asic/projects/C/CMS_PIX_28/benjamin/testing/workarea_112024/CMSPIX28_DAQ/spacely/PySpacely/calibrationNPix.npy', matrix)

pattern = np.tile([1, 1, 0], 256)  # Create the repeating pattern: [1, 1, 0] * 256
matches = np.all(matrix == pattern, axis=1)

# If there is an exact match, print the indices
if np.any(matches):
    print("Exact match found at indices:", np.where(matches)[0])
else:
    print("No exact match found.")

    # Step 2: If no exact match, find the closest match using Euclidean distance
    # Calculate the Euclidean distance between the pattern and each row in the array
    distances = np.linalg.norm(matrix - pattern, axis=1)
    
    # Find the index of the row with the smallest distance
    closest_index = np.argmin(distances)

    print(f"Closest match found at index: {closest_index}")
    print(f"Distance: {distances[closest_index]}")