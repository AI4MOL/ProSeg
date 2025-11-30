import torch
import numpy as np
from dataloader.data_loader import DatasetSpliter
import nibabel as nib
import cv2
import os
import argparse
from tqdm import tqdm

from configs.config import *
from utils.utils import rand_seed, show_img
from dataloader.utils import data_preprocess
from lib.metrics_set import *
from lib.initialize_model import init_model

def evaluate(net, test_loader, opt, result_path):
    GED_global, Dice_max, Dice_max_reverse, Dice_soft, Dice_match, Dice_each, cd, d_0, d_1  = 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0] * 4), 0.0, 0.0, 0.0

    net.eval()
    with torch.no_grad():
        for test_step, (patch, masks, sid) in enumerate(tqdm(test_loader)):

            patch, masks = data_preprocess(patch, masks, training=False)
            patch = patch.cuda()

            outputs = net.test_step(patch)
            preds = torch.sigmoid(outputs).cpu()

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

            if opt.VISUALIZE:
                concat_pred = show_img(patch, preds, masks)
                cv2.imshow('predictions', concat_pred)
                cv2.waitKey(0)
            
            if opt.TEST_SAVE:
                patch = patch.cpu().numpy()
                masks = masks.numpy()
                preds = preds.numpy()
                nib.save(nib.Nifti1Image(patch[0,0].astype(np.float32), np.eye(4)), result_path +  "%02d_image.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,0].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a1.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,1].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a2.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,2].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a3.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,3].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a4.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,0]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s1.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,1]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s2.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,2]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s3.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,3]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s4.nii.gz" % test_step)

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
    parser.add_argument("--config", type=str, default='./configs/params_lidc.yaml', help="config path (*.yaml)")
    parser.add_argument("--save_path", type=str, default='../models/pionono_lidc_20231101-210220/', help="save path")
    parser.add_argument("--model_name", type=str, default='pionono')
    parser.add_argument("--mask_num", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    opt = Config(config_path=args.config)
    rand_seed(opt.RANDOM_SEED)

    # dataset
    data_spliter = DatasetSpliter(opt=opt, input_size=opt.INPUT_SIZE)
    evaluate_records = []

    for fold_idx in range(opt.KFOLD):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        print('#********{} of {} FOLD *******#'.format(fold_idx+1, opt.KFOLD))
        train_loader, test_loader = data_spliter.get_datasets(fold_idx=fold_idx)
        rand_seed(opt.RANDOM_SEED)

        net = init_model(args, opt)

        ckpt = torch.load(os.path.join(args.save_path,'{}_{}_{}_best.pth'.format(args.model_name, opt.DATASET, fold_idx)))
        net.load_state_dict(ckpt['model'])

        net.cuda()

        result_path = args.save_path + 'results_prior_{}_fold/'.format(fold_idx)
        os.makedirs(result_path, exist_ok=True)

        metrics_dict = evaluate(net, test_loader, opt, result_path)
        evaluate_records.append(metrics_dict)
        for key in metrics_dict.keys():
            print(key, ': ', metrics_dict[key])

    print(args.save_path)
    with open(args.save_path+'performance_prior.txt', 'w') as f:
        for key in evaluate_records[0].keys():
            temp = []
            for record in evaluate_records:
                temp.append(record[key])
            print('{}: {}±{}'.format(key, np.mean(temp, axis=0), np.std(temp, axis=0, ddof=0)))
            f.writelines('{}: {}±{} \n'.format(key, np.mean(temp, axis=0), np.std(temp, axis=0, ddof=0)))