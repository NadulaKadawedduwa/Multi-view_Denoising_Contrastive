import numpy as np
import cv2
import os
import re

# get folder names and sort 
current_directory = 'W:\\ExposureTest\\'
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
    file_path = current_directory + folder + '/RAWimages/RAWImageBoard106Camera1.RAW'
    # Read the raw data into a numpy array
    raw_data = np.fromfile(file_path, dtype=np.uint16)
    # Reshape the data based on the resolution (4032x3040)
    raw_array = raw_data.reshape((3040, 4032))
    raw_array_list.append(raw_array)
    R_channel = raw_array[::2, ::2]  # Select every other pixel starting from the top-left
    G_channel1 = raw_array[::2, 1::2]  # Select every other pixel starting from the second pixel in the first row
    G_channel2 = raw_array[1::2, ::2]  # Select every other pixel starting from the second pixel in the second row  
    B_channel = raw_array[1::2, 1::2] # Select every other pixel starting from the second pixel in the second row
    # G_channel = (G_channel1 + G_channel2) / 2
    G_channel = G_channel1
    R_channel_list.append(R_channel)
    G_channel_list.append(G_channel)
    B_channel_list.append(B_channel)
    # cv2.imwrite('ExtractedChannelImages\\' + folder + '_Red.png', R_channel.astype(np.uint16))
    # cv2.imwrite('ExtractedChannelImages\\' + folder + '_Green.png', G_channel.astype(np.uint16))
    # cv2.imwrite('ExtractedChannelImages\\' + folder + '_Blue.png', B_channel.astype(np.uint16))


GT_R_channel = np.zeros_like(R_channel_list[0], dtype=np.float64)
GT_G_channel = np.zeros_like(G_channel_list[0], dtype=np.float64)
GT_B_channel = np.zeros_like(B_channel_list[0], dtype=np.float64)


# average region of interest

# ROI half resolution
top_left = (1052, 639)  
bottom_right = (1100, 691)  

# ROI full resolution
top_left = (2013, 1277)  
bottom_right = (2199, 1384)  

for index, image in enumerate(raw_array_list):
    roi = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    average_color = np.mean(roi)
    print(str(index) + ": Average pixel values ROI:", average_color)


# CV2 color checker detection

calibImagePath = current_directory + "ColorCheckerExp40000000_1/RAWImagesPNG/Board106Camera1.png"
image = cv2.imread(calibImagePath)
detector = cv2.mcc.CCheckerDetector_create()
detector.process(image, cv2.mcc.MCC24, 1)

checkers = detector.getListColorChecker()
checker = checkers[0]

cdraw = cv2.mcc.CCheckerDraw_create(checker)
img_draw = image.copy()
cdraw.draw(img_draw)

cv2.imwrite("RecognizedChecker.png", img_draw)


# debugging & development scripts

for i in range(5,15):
    GT_R_channel += R_channel_list[i] / 10.0
    GT_G_channel += G_channel_list[i] / 10.0
    GT_B_channel += B_channel_list[i] / 10.0

for im in G_channel_list:
    im[1080][735]

GT_G_channel[1080][735]

cv2.imwrite('GT_R.png', GT_R_channel.astype(np.uint16))
cv2.imwrite('GT_G.png', GT_G_channel.astype(np.uint16))
cv2.imwrite('GT_B.png', GT_B_channel.astype(np.uint16))




# draw chart comparing pixel values correlated with exposure times
import matplotlib.pyplot as plt

# Numbers to plot
numbers = [18579, 6790, 3267, 2369, 2114, 2051]
exposure_times = [40000000, 10000000, 2500000, 625000, 156250, 39062]

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(exposure_times, numbers, marker='o', linestyle='-')
ax.set_title('Pixel (2166, 1465) Comparison Of Multiple Exposures')
ax.set_xlabel('Exposure time in nanoseconds')
ax.set_ylabel('16bit pixel value')
ax.grid(True)

plt.show()