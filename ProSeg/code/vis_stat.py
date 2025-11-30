import numpy as np
from tqdm import tqdm
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from lib.metrics_set import  distance_one as distance
import random


def get_data(path):
    nii_file = nib.load(path)
    data = nii_file.get_fdata()
    #print(data.shape)
    if len(data.shape) == 4:
        data = data.reshape(-1, 128, 128)
    elif len(data.shape) == 2:
        data = data.reshape(1, 128, 128)
    return data

image_ids = []
for i in range(2):
    for j in range(10):
        image_ids.append(str(i)+str(j))
        
root_path =  "/data/D-Persona/models/pionono_prob01_LIDC_20250118-032137results_0_fold"

rank_list = []


for image_id in tqdm(image_ids):
    image_path_1 = f'{root_path}/{image_id}_pred_s1.nii.gz'
    image_path_2 = f'{root_path}/{image_id}_pred_s2.nii.gz'
    image_path_3 = f'{root_path}/{image_id}_pred_s3.nii.gz'
    image_path_4 = f'{root_path}/{image_id}_pred_s4.nii.gz'
    image_data_1 = get_data(image_path_1)
    image_data_2 = get_data(image_path_2)
    image_data_3 = get_data(image_path_3)
    image_data_4 = get_data(image_path_4)
    label_path_1 = f'{root_path}/{image_id}_label_a1.nii.gz'
    label_path_2 = f'{root_path}/{image_id}_label_a2.nii.gz'
    label_path_3 = f'{root_path}/{image_id}_label_a3.nii.gz'
    label_path_4 = f'{root_path}/{image_id}_label_a4.nii.gz'
    label_data_1 = get_data(label_path_1)
    label_data_2 = get_data(label_path_2)
    label_data_3 = get_data(label_path_3)
    label_data_4 = get_data(label_path_4)
    num_slices = image_data_1.shape[0]
    #print(num_slices, image_data_1.shape, label_data_1.shape)
    for one_slice in range(num_slices):
        image_1 = image_data_1[one_slice].astype(np.int32)
        image_2 = image_data_2[one_slice].astype(np.int32)
        image_3 = image_data_3[one_slice].astype(np.int32)
        image_4 = image_data_4[one_slice].astype(np.int32)
        label_1 = label_data_1[one_slice].astype(np.int32)
        label_2 = label_data_2[one_slice].astype(np.int32)
        label_3 = label_data_3[one_slice].astype(np.int32)
        label_4 = label_data_4[one_slice].astype(np.int32)
        distance_1 = distance(image_1, label_1)
        distance_2 = distance(image_2, label_2)
        distance_3 = distance(image_3, label_3)
        distance_4 = distance(image_4, label_4)
        average_distance_aggre = (distance_1 + distance_2 + distance_3 + distance_4) / 4.0
        
        average_distance = (distance(image_1, image_2)+
                            distance(image_1, image_3)+
                            distance(image_1, image_4)+
                            distance(image_2, image_3)+
                            distance(image_2, image_4)+
                            distance(image_3, image_4))/6.0
        
        dict_rank = {"image_id": image_id, "slice": one_slice, "average_distance": average_distance-average_distance_aggre}
        rank_list.append(dict_rank)

rank_list = sorted(rank_list, key=lambda x: x["average_distance"], reverse=True)
print(rank_list[:20])