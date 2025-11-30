import shutil
import os
import glob

path1 = '/mnt/nas/share2/data/MedImage/training'
path2 = '/mnt/nas/share2/data/MedImage/DP/validation'

if not os.path.exists(path2):
    os.makedirs(path2)

for i in range(80,100):
    for src in glob.glob(os.path.join(path1, f'Sample_{i}*')):
        filename = os.path.basename(src)
        dst = os.path.join(path2, filename)
        shutil.move(src, dst)
