import numpy as np
from tqdm import tqdm
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from lib.metrics_set import  distance_one as distance
import random
import json

image_ids = []
for i in range(2):
    for j in range(10):
        image_ids.append(str(i)+str(j))


# dis_list = [{'image_id': '06', 'slice': 20, 'average_distance': np.array([0.12891139])}, {'image_id': '15', 'slice': 19, 'average_distance': np.array([0.13200621])}, {'image_id': '19', 'slice': 18, 'average_distance': np.array([0.13536647])},  {'image_id': '12', 'slice': 20, 'average_distance': np.array([0.17205665])}]

# dis_list = [{'image_id': '10', 'slice': 10, 'average_distance': np.array([0.12891139])}, {'image_id': '14', 'slice': 13, 'average_distance': np.array([0.13200621])}, {'image_id': '17', 'slice': 8, 'average_distance': np.array([0.13536647])},  {'image_id': '19', 'slice': 8, 'average_distance': np.array([0.17205665])}]
#########################LIDC############################################
dis_list = [{'image_id': '10', 'slice': 0, 'average_distance': np.array([0.04854335])}, {'image_id': '01', 'slice': 0, 'average_distance': np.array([0.03607119])}, {'image_id': '08', 'slice': 0, 'average_distance': np.array([0.02749731])}, {'image_id': '14', 'slice': 0, 'average_distance': np.array([0.02520134])}]

target_path = "/data/D-Persona/models/pionono_prob01_LIDC_20250118-032137results_0_fold"
root_paths = [
    target_path,
    "/data/D-Persona/models/cm_global01_LIDC_20250125-112504results_0_fold",
    "/data/D-Persona/models/cm_pixel01_LIDC_20250125-130753results_0_fold",
    "/data/D-Persona/models/pionono_mix_LIDC_20250113-233642results_0_fold",
    "/data/D-Persona/models/pionono_prob01_LIDC_20250118-032137results_prior_0_fold",
    "/data/D-Persona/models/prob_unet01_LIDC_20250125-072907results_0_fold",
    "/data/D-Persona/models/prob_unet01_LIDC_20250125-072907results_2_fold"
]

method_names = ["pionono_prob_lidc",
              "cm_global_lidc",
              "cm_pixel_lidc",
              "mix_lidc",
              "proseg_prior_lidc",
              "probunet_lidc",
              "final"
              ]
#########################NPC############################################
# dis_list = [{'image_id': '10', 'slice': 10, 'average_distance': np.array([0.12891139])}, {'image_id': '14', 'slice': 13, 'average_distance': np.array([0.13200621])}, {'image_id': '17', 'slice': 8, 'average_distance': np.array([0.13536647])},  {'image_id': '19', 'slice': 8, 'average_distance': np.array([0.17205665])}]

# target_path =  "/data/D-Persona/models/DPersona2_NPC_20250115-050904results"

# root_paths = [
#             "/mnt/nas/share2/home/liuke/edata/DPersona1_NPC_20250114-232424results",
#               "/mnt/nas/share2/home/liuke/edata/DPersona2_NPC_20250115-050904results",
#               "/mnt/nas/share2/home/liuke/edata/cm_global01_2_NPC_20250122-103651results",
#               "/mnt/nas/share2/home/liuke/edata/cm_pixel01_2_NPC_20250122-122047results",
#               "/mnt/nas/share2/home/liuke/edata/prob_unet01_2_NPC_20250122-065829results",
#               "/mnt/nas/share2/home/liuke/edata/TAB-232522results",
#               "/mnt/nas/share2/home/liuke/edata/pionono01_NPC_20250115-001034results",
#               "/mnt/nas/share2/home/liuke/edata/pionono_prob01_2_NPC_20250118-012413results",
#               "/data/D-Persona/models/pionono_prob01_2_NPC_20250118-012413results_prior"
#               ]
# method_names = ["DPersona1",
#               "DPersona2",
#               "cm_global01_2",
#               "cm_pixel01_2",
#               "prob_unet01_2",
#               "TAB",
#               "pionono01",
#               "pionono_prob",
#               "pionono_prob_prior"
#               ]
def get_data(path):
    nii_file = nib.load(path)
    data = nii_file.get_fdata()
    #print(data.shape)
    if len(data.shape) == 4:
        data = data.reshape(-1, 128, 128)
    elif len(data.shape) == 2:
        data = data.reshape(1, 128, 128)
    return data
rank_list = []

for one_root_path in root_paths:
    for one_dict in dis_list:
        FLAG = True
        # print(one_dict)
        for image_id in tqdm(image_ids):
            label_path_1 = f'{one_root_path}/{image_id}_label_a1.nii.gz'
            label_path_2 = f'{one_root_path}/{image_id}_label_a2.nii.gz'
            label_path_3 = f'{one_root_path}/{image_id}_label_a3.nii.gz'
            label_path_4 = f'{one_root_path}/{image_id}_label_a4.nii.gz'
            label_data_1 = get_data(label_path_1)
            label_data_2 = get_data(label_path_2)
            label_data_3 = get_data(label_path_3)
            label_data_4 = get_data(label_path_4)
            #'image_id': '06', 'slice': 20, 'average_distance'
            target_label_data_1 = get_data(f'{target_path}/{one_dict["image_id"]}_label_a1.nii.gz')
            target_label_data_2 = get_data(f'{target_path}/{one_dict["image_id"]}_label_a2.nii.gz')
            target_label_data_3 = get_data(f'{target_path}/{one_dict["image_id"]}_label_a3.nii.gz')
            target_label_data_4 = get_data(f'{target_path}/{one_dict["image_id"]}_label_a4.nii.gz')
            target_slice = one_dict["slice"]
            if target_slice > label_data_1.shape[0]:
                continue
            target_label_1 = target_label_data_1[target_slice].astype(np.int32)
            target_label_2 = target_label_data_2[target_slice].astype(np.int32)
            target_label_3 = target_label_data_3[target_slice].astype(np.int32)
            target_label_4 = target_label_data_4[target_slice].astype(np.int32)
            source_label_1 = label_data_1[target_slice].astype(np.int32)
            source_label_2 = label_data_2[target_slice].astype(np.int32)
            source_label_3 = label_data_3[target_slice].astype(np.int32)
            source_label_4 = label_data_4[target_slice].astype(np.int32)
            distance_1 = distance(source_label_1, target_label_1)
            distance_2 = distance(source_label_2, target_label_2)
            distance_3 = distance(source_label_3, target_label_3)
            distance_4 = distance(source_label_4, target_label_4)
            average_distance = (distance_1 + distance_2 + distance_3 + distance_4) / 4.0
            if average_distance < 0.001:
                FLAG =  False
                # print(one_root_path, image_id, target_slice, average_distance)
                temp_dict = {"image_id": image_id, "slice": target_slice, "root_path": one_root_path}
                rank_list.append(temp_dict)
                print(one_root_path, average_distance)
                break
        if FLAG:
            print(one_root_path, "No match")
            temp_dict = {"image_id": one_dict["image_id"], "slice": one_dict["slice"], "root_path": one_root_path}
            rank_list.append(temp_dict)

output_path = "/data/D-Persona/code/rank_list.json"
with open(output_path, 'w') as f:
    json.dump(rank_list, f, indent=4)
    
        

