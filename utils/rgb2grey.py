import cv2
import os
from skimage import color
from skimage import io
#color의 이미지를 grey로 변경 lab 중에 lchannel만 저장

origin_data_paths = os.path.join("srdata/" "color")
new_data_paths = os.path.join("srdata/" "grey")

origin_filenames = os.listdir(origin_data_paths)


origin_full_filenames = [os.path.join(origin_data_paths, f) for f in origin_filenames]
new_full_filenames = [os.path.join(new_data_paths, f) for f in origin_filenames]

for idx in range(len(origin_full_filenames)):
    image=io.imread(origin_full_filenames[idx])
    lchannel, achannel, bchannel = cv2.split(color.rgb2lab(image))
    io.imsave(new_full_filenames[idx],lchannel)








