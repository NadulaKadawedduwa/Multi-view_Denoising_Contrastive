import numpy as np
import imageio
import os
import re

# get folder names and sort 
current_directory = os.getcwd()
items = os.listdir(current_directory)
folders = [item for item in items if os.path.isdir(os.path.join(current_directory, item)) and re.match(r'^ColorCheckerExp\d+(?:_\d+)?$', item)]
folders_sorted = sorted(folders, key=lambda x: [int(num) for num in re.findall(r'\d+', x)])

raw_array_list = []
R_channel_list = []
G_channel_list = []
B_channel_list = []

for index, folder in enumerate(folders_sorted):
    print(index, folder)
    # Extract the center image
    file_path = folder + '/RAWimages/RAWImageBoard106Camera1.RAW'
    
    # Read the raw data into a numpy array
    raw_data = np.fromfile(file_path, dtype=np.uint16)

    # Reshape the data based on the resolution (4032x3040)
    raw_array = raw_data.reshape((3040, 4032))
    raw_array_list.append(raw_array)
    R_channel = raw_array[::2, ::2]  # Select every other pixel starting from the top-left
    G_channel = raw_array[::2, 1::2] # Select every other pixel starting from the second pixel in the first row
    B_channel = raw_array[1::2, 1::2]# Select every other pixel starting from the second pixel in the second row
    R_channel_list.append(R_channel)
    G_channel_list.append(G_channel)
    B_channel_list.append(B_channel)

    # imageio.imwrite('ExtractedChannelImages\\' + folder + '_Red.png', R_channel.astype(np.uint16), compression='png', compression_level=0)
    # imageio.imwrite('ExtractedChannelImages\\' + folder + '_Green.png', G_channel.astype(np.uint16), compression='png', compression_level=0)
    # imageio.imwrite('ExtractedChannelImages\\' + folder + '_Blue.png', B_channel.astype(np.uint16), compression='png', compression_level=0)


