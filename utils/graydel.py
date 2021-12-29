import os
import shutil
import re
import imageio
#train 에서 그레이 이미지를 outlier로 이동
data_paths = "srdata/train"
temp="srdata/outliers"
filenames = os.listdir(data_paths)
full_filenames=[os.path.join(data_paths,f) for f in filenames]
for _ in range(len(full_filenames)):
        image= imageio.imread(full_filenames[_])

        if len(image.shape)==2:

                shutil.move(full_filenames[_],re.sub('train','outliers',full_filenames[_]))


