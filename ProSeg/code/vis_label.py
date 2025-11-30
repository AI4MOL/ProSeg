import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import SimpleITK as sitk
from skimage import measure
from find_exact_match import dis_list, target_path, method_names
import cv2

def morphology_denoise(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned_mask


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

def close_small_gaps(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed_mask



#SLICE = 8
def load_image(image_path):
    image_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    if len(image_arr.shape) == 4:
        image_arr = image_arr.reshape(128, 128, -1)
    elif len(image_arr.shape) == 2:
        image_arr = image_arr.reshape(128, 128, 1)
    #print(image_arr.shape)
    return image_arr

def load_mask(mask_path):
    maks_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    #print(maks_array.shape)
    if len(maks_array.shape) == 4:
        maks_array = maks_array.reshape(128, 128, -1)
    elif len(maks_array.shape) == 2:
        maks_array = maks_array.reshape(128, 128, 1)
    return maks_array


# Path to the JSON file
file_path = "rank_list.json"

#target_path =  "/data/D-Persona/models/pionono_prob01_2_NPC_20250118-012413results_prior"
#dis_list = [{'image_id': '06', 'slice': 20, 'average_distance': np.array([0.12891139])}, {'image_id': '15', 'slice': 19, 'average_distance': np.array([0.13200621])}, {'image_id': '19', 'slice': 18, 'average_distance': np.array([0.13536647])},  {'image_id': '12', 'slice': 20, 'average_distance': np.array([0.17205665])}]

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

image_paths = []
for one_dict in dis_list:
    image_id = one_dict["image_id"]
    # image_paths.append(f'{target_path}/{image_id}_image_t2.nii.gz')
    # image_paths.append(f'{target_path}/{image_id}_image_t2.nii.gz')
    # image_paths.append(f'{target_path}/{image_id}_image_t2.nii.gz')
    # image_paths.append(f'{target_path}/{image_id}_image_t2.nii.gz')
    image_paths.append(f'{target_path}/{image_id}_image.nii.gz')
    image_paths.append(f'{target_path}/{image_id}_image.nii.gz')
    image_paths.append(f'{target_path}/{image_id}_image.nii.gz')
    image_paths.append(f'{target_path}/{image_id}_image.nii.gz')

def get_pics(image_paths, mask_paths, slices, name):
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    colors = ['r', 'g', 'b', 'y']  # 定义不同的颜色

    for i in range(4):
        print(load_image(image_paths[i]).shape)
        image = load_image(image_paths[i])[:,:,slices[i]]
        masks = []
        for ind, mask_path in enumerate(mask_paths[i]):
            masks.append(close_small_gaps(morphology_denoise(load_mask(mask_path)[:,:,slices[i]])))
        # 显示图像
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')  # 关闭坐标轴
        
        # 显示每个mask
        for j in range(4):
            contours = measure.find_contours(masks[j], 0.5)
            for contour in contours:
                axes[i].plot(contour[:, 1], contour[:, 0], 
                            color=colors[j], 
                            linewidth=4, 
                            label=f'Mask {j+1}' if i == 0 else "")  # 只在第一个子图保留图例
        

    handles = []
    labels = []
    for handle, label in zip(*axes[1].get_legend_handles_labels()):
        if label not in labels:  # 避免重复
            handles.append(handle)
            labels.append(label)
    axes[3].legend(handles, labels,loc='upper right')
    plt.tight_layout()
    plt.savefig(f'vis/save_final/{name}.png', bbox_inches='tight')
    plt.close(fig)

count = 0
name_index = 0
while count < len(data):
    
    mask_paths = []
    slices = []
    root_path = data[count]["root_path"]
    for i in range(4):
        image_id = data[count+i]["image_id"]
        slice = data[count+i]["slice"]
        slices.append(slice)
        temp_paths = [
                f'{root_path}/{image_id}_pred_s1.nii.gz',
                f'{root_path}/{image_id}_pred_s2.nii.gz',
                f'{root_path}/{image_id}_pred_s3.nii.gz',
                f'{root_path}/{image_id}_pred_s4.nii.gz'
            ]
        mask_paths.append(temp_paths)
    get_pics(image_paths, mask_paths, slices, method_names[name_index])
    count += 4
    name_index += 1
    print(f'{name_index}/{len(method_names)}')
    if name_index == len(method_names):
        break

label_mask_paths = []
slice = []
for one_dict in dis_list:
    image_id = one_dict["image_id"]
    temp_mask = [f'{target_path}/{image_id}_label_a1.nii.gz', f'{target_path}/{image_id}_label_a2.nii.gz', f'{target_path}/{image_id}_label_a3.nii.gz', f'{target_path}/{image_id}_label_a4.nii.gz']
    label_mask_paths.append(temp_mask)
    slice.append(one_dict["slice"])
get_pics(image_paths, label_mask_paths, slice, "Gold")
