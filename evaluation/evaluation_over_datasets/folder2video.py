import os
import sys
import cv2
import collections

path = sys.argv[1]
print("Scanning {}".format(path))

sorted_dict = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.png'):
            filename_cnt = filename.split('_')[0]
            sorted_dict[int(filename_cnt)] = os.sep.join([dirpath, filename])

sorted_dict = collections.OrderedDict(sorted(sorted_dict.items()))


# choose codec according to format needed
example_file_name = min(sorted_dict.keys())
height, width, C = cv2.imread(sorted_dict[example_file_name]).shape
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter('video.mp4', fourcc, 5, (width, height))

for k, v in sorted_dict.items():
  img = cv2.imread(v)
  video.write(img)

cv2.destroyAllWindows()
video.release()
