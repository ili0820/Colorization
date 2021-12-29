import numpy
import math
import cv2
import os
#test와 result안의 이미지들의 psnr 계산
data_type=["test","result"]

origin_data_paths = os.path.join("srdata/" + data_type[0])
result_data_paths = os.path.join("output/" + data_type[1])

origin_filenames = os.listdir(origin_data_paths)
result_filenames = os.listdir(result_data_paths)

origin_full_filenames = [os.path.join(origin_data_paths, f) for f in origin_filenames]
result_full_filenames = [os.path.join(result_data_paths, f) for f in result_filenames]

def psnr(img1, img2):

    mse = numpy.mean( (img1 - img2) ** 2 ) #MSE 구하는 코드

    # print("mse : ",mse)

    if mse == 0:

        return 100

    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR구하는 코드
d=[]
for org,res in zip(origin_full_filenames,result_full_filenames):
    origin=cv2.imread(org)
    result=cv2.imread(res)


    d.append(psnr(origin,result))
print("AVG PSNR:",sum(d)/len(d))
print("MAX PSNR:",max(d))
print("MIN PSNR:",min(d))


