import torch
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import nibabel as nib
from configs.config import *
from utils.utils import rand_seed, show_img
from lib.metrics_set import *
from dataloader.dataset import BaseDataSets, ZoomGenerator
from torch.utils.data import DataLoader
from lib.initialize_model import init_model
import random
random_seed = random.randint(1, 10000)

def validate(net, val_loader, opt, writer=None, times_step = 0, all_metrics=False):
    GED_global, Dice_max, Dice_soft, cd, d_0, d_1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    net.eval()
    with torch.no_grad():
        for val_step, sample in enumerate(tqdm(val_loader)):

            patch = sample['image'].cuda()
            masks = sample['label'].float()

            preds = []
            for idx in range(patch.shape[2]):
                output_slice = net.val_step(patch[:,:,idx]).unsqueeze(2)
                output_slice = torch.sigmoid(output_slice).cpu()
                preds.append(output_slice)
            preds = torch.cat(preds, 2)
            # Dice score
            GED_iter, cd_iter, d_0_iter, d_1_iter = generalized_energy_distance(masks, preds, all=all_metrics)
            dice_max_iter, dice_max_reverse_iter, _, _ = dice_at_all(masks, preds, thresh=0.5)
            dice_soft_iter = dice_at_thresh(masks, preds)

            cd += cd_iter
            d_0 += d_0_iter
            d_1 += d_1_iter
            GED_global += GED_iter
            Dice_max += (dice_max_iter + dice_max_reverse_iter) / 2
            Dice_soft += dice_soft_iter

            index_z = preds.shape[2] // 2
            if opt.VISUALIZE:
                concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
                cv2.imshow('predictions', concat_pred)
                cv2.waitKey(0)
            
            if writer is not None and val_step == len(val_loader) // 2:
                concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
                writer.add_image('Images', concat_pred, times_step, dataformats='HW')

    # store in dict
    metrics_dict = {'GED': GED_global / len(val_loader),
                    'Dice_max': Dice_max / len(val_loader),
                    'Dice_soft': Dice_soft / len(val_loader)}
    if all_metrics:
        metrics_dict['GED_cross'] = cd / len(val_loader)
        metrics_dict['GED_d0'] = d_0 / len(val_loader)
        metrics_dict['GED_d1'] = d_1 / len(val_loader)
    return metrics_dict

def validate_unet(net, val_loader, opt, writer=None, times_step = 0, all_metrics=False):
    GED_global, Dice_max, Dice_soft, cd, d_0, d_1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    net.eval()
    with torch.no_grad():
        for val_step, sample in enumerate(tqdm(val_loader)):

            patch = sample['image'].cuda()
            masks = sample['label'].float()
            print(masks.shape)
            preds = []
            for idx in range(patch.shape[2]):
                output_slice = net.val_step(patch[:,:,idx]).unsqueeze(2)
                output_slice = torch.sigmoid(output_slice).cpu()
                preds.append(output_slice)
            preds = torch.cat(preds, 2)
            # Dice score
            GED_iter, cd_iter, d_0_iter, d_1_iter = generalized_energy_distance(masks, preds, all=all_metrics)
            #dice_max_iter, dice_max_reverse_iter, _, _ = dice_at_all(masks, preds, thresh=0.5)
            dice_soft_iter = dice_at_thresh(masks, preds)

            cd += cd_iter
            d_0 += d_0_iter
            d_1 += d_1_iter
            GED_global += GED_iter
            #Dice_max += (dice_max_iter + dice_max_reverse_iter) / 2
            Dice_soft += dice_soft_iter

            # index_z = preds.shape[2] // 2
            # if opt.VISUALIZE:
            #     concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
            #     cv2.imshow('predictions', concat_pred)
            #     cv2.waitKey(0)
            
            # if writer is not None and val_step == len(val_loader) // 2:
            #     concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
            #     writer.add_image('Images', concat_pred, times_step, dataformats='HW')

    # store in dict
    metrics_dict = {'GED': GED_global / len(val_loader),
                    #'Dice_max': Dice_max / len(val_loader),
                    'Dice_soft': Dice_soft / len(val_loader)}
    if all_metrics:
        metrics_dict['GED_cross'] = cd / len(val_loader)
        metrics_dict['GED_d0'] = d_0 / len(val_loader)
        metrics_dict['GED_d1'] = d_1 / len(val_loader)
    return metrics_dict


def evaluate(net, test_loader, opt, result_path):
    GED_global, Dice_max, Dice_max_reverse, Dice_soft, Dice_match, Dice_each, cd, d_0, d_1  = 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0] * 4), 0.0, 0.0, 0.0

    net.eval()
    with torch.no_grad():
        for test_step, sample in enumerate(tqdm(test_loader)):

            patch = sample['image'].cuda()
            masks = sample['label'].float()

            preds = []
            for idx in range(patch.shape[2]):
                output_slice = net.val_step(patch[:,:,idx]).unsqueeze(2)
                output_slice = torch.sigmoid(output_slice).cpu()
                preds.append(output_slice)
            preds = torch.cat(preds, 2)
            
            GED_iter, cd_iter, d_0_iter, d_1_iter = generalized_energy_distance(masks, preds, all=True)
            # Dice score
            dice_max_iter, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(masks, preds, thresh=0.5)
            dice_soft_iter = dice_at_thresh(masks, preds)

            cd += cd_iter
            d_0 += d_0_iter
            d_1 += d_1_iter
            GED_global += GED_iter
            Dice_match += dice_match_iter
            Dice_max += dice_max_iter
            Dice_max_reverse += dice_max_reverse_iter
            Dice_soft += dice_soft_iter
            Dice_each += np.array(dice_each_iter)

            # index_z = preds.shape[2] // 2
            # if opt.VISUALIZE:
            #     concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
            #     cv2.imshow('predictions', concat_pred)
            #     cv2.waitKey(0)
            
            # if opt.TEST_SAVE:
            #     patch = patch.cpu().numpy()
            #     masks = masks.numpy()
            #     preds = preds.numpy()
            #     nib.save(nib.Nifti1Image(patch[0,0].astype(np.float32), np.eye(4)), result_path +  "%02d_image_t1.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image(patch[0,1].astype(np.float32), np.eye(4)), result_path +  "%02d_image_t1c.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image(patch[0,2].astype(np.float32), np.eye(4)), result_path +  "%02d_image_t2.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image(masks[0,0].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a1.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image(masks[0,1].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a2.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image(masks[0,2].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a3.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image(masks[0,3].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a4.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image((preds[0,0]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s1.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image((preds[0,1]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s2.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image((preds[0,2]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s3.nii.gz" % test_step)
            #     nib.save(nib.Nifti1Image((preds[0,3]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s4.nii.gz" % test_step)

    # store in dict
    metrics_dict = {'GED': GED_global / len(test_loader),
                    'Dice_max': Dice_max / len(test_loader),
                    'Dice_max_reverse': Dice_max_reverse / len(test_loader),
                    'Dice_max_mean': (Dice_max_reverse + Dice_max) / (2 * len(test_loader)),
                    'Dice_match': Dice_match / len(test_loader),
                    'Dice_soft': Dice_soft / len(test_loader),
                    'Dice_each': Dice_each / len(test_loader),
                    'Dice_each_mean': np.mean(Dice_each) / len(test_loader)}
    metrics_dict['GED_cross'] = cd / len(test_loader)
    metrics_dict['GED_d0'] = d_0 / len(test_loader)
    metrics_dict['GED_d1'] = d_1 / len(test_loader)
    
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/params_npc.yaml', help="config path (*.yaml)")
    parser.add_argument("--save_path", type=str, default='../models/pionono_prob01_2_NPC_20250119-045143/', help="save path")
    parser.add_argument("--model_name", type=str, default='pionono_prob')
    parser.add_argument("--mask_num", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--latent_dim", type=int, default=8)
    args = parser.parse_args()
    
    models = ['DPersona', 'prob_unet', 'pionono', 'pionono_mix', 'pionono_prob', 'cm_global', 'cm_pixel']
    # models = ['pionono_prob']
    for model_name in models:
        args.model_name = model_name
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        opt = Config(config_path=args.config)
        rand_seed(random_seed)
        net = init_model(args, opt)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"总参数量 of {args.model_name}:", total_params)

        # 可训练参数量
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"可训练参数量 of {args.model_name}:", trainable_params)
        # print(net)

        def count_filtered_params(model, ignore_keywords=['reconstruction_head', 'classifier']):
            total = 0
            flag = 0
            for name, param in model.named_parameters():
                # if model_name == 'pionono_prob':
                #     if any(k in name for k in ignore_keywords):
                #         if flag == 0:
                #             print(f"排除参数: {name}")
                #             flag = 1
                #         # print(f"排除参数: {name}")
                #         continue
                if model_name == 'DPersona':
                    if 'posterior' in name:
                        if flag == 0:
                            print(f"排除参数: {name}")
                            flag = 1
                        continue
                total += param.numel()
            return total

        def format_params(n):
            if n >= 1e12:
                return f"{n / 1e12:.2f}T"
            elif n >= 1e9:
                return f"{n / 1e9:.2f}B"
            elif n >= 1e6:
                return f"{n / 1e6:.2f}M"
            elif n >= 1e3:
                return f"{n / 1e3:.2f}K"
            return str(n)

        # ===============================
        # 执行统计
        # ===============================

        filtered_param_count = count_filtered_params(net)
        formatted = format_params(filtered_param_count)

        print(f"\n去除包含 'decoder' 或 'classifier' 的参数后，总参数量为：{formatted}")