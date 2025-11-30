import torch

from tensorboardX import SummaryWriter

import os
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import shutil
import random
from configs.config import *
from evaluate_npc_unet import validate_unet as validate
from utils.logger import Logger
from utils.utils import rand_seed
from dataloader.dataset import RandomGenerator_Multi_Rater, BaseDataSets, ZoomGenerator
from torch.utils.data import DataLoader
from lib.initialize_model import init_model
from lib.initialize_optimization import init_optimization

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/params_npc.yaml', help="config path (*.yaml)")
    parser.add_argument("--save_path", type=str, help="save path", default='')
    parser.add_argument("--model_name", type=str, default='unet') #[prob_unet, cm_global, cm_pixel, pionono]
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--mask_num", type=int, default=4)
    parser.add_argument("--RESUME_FROM", type=int, default=0)
    parser.add_argument("--kl", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=2)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--annotater", type=int, default=0)
    
    args = parser.parse_args()
    opt = Config(config_path=args.config)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    rand_seed(opt.RANDOM_SEED)

    # log & model folder
    if args.save_path == '':
        opt.MODEL_DIR += args.model_name+f'A_{args.annotater}' +'_{}_{}'.format(opt.DATASET, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        opt.MODEL_DIR = args.save_path

    if not os.path.exists(opt.MODEL_DIR):
        os.mkdir(opt.MODEL_DIR)

    logger = Logger(args.model_name, path=opt.MODEL_DIR)
    writer = SummaryWriter(opt.MODEL_DIR)

    shutil.copytree('../code/', opt.MODEL_DIR + '/code/', shutil.ignore_patterns(['.git','__pycache__']))

     # dataset
    db_train = BaseDataSets(
        base_dir=opt.DATA_PATH,
        split="train",
        transform=RandomGenerator_Multi_Rater(opt.PATCH_SIZE)
    )

    db_val = BaseDataSets(
        base_dir=opt.DATA_PATH,
        split="val",
        transform=ZoomGenerator(opt.PATCH_SIZE)
    )
    def worker_init_fn(worker_id):
        random.seed(opt.RANDOM_SEED + worker_id)

    train_loader = DataLoader(db_train, batch_size=opt.TRAIN_BATCHSIZE, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=True, num_workers=1)

    # Training Config
    epochs = args.epochs
    epoch_start = 0

    net = init_model(args, opt, kl=args.kl, sigma=args.sigma, latent_dim=args.latent_dim)
    #for name, param in net.named_parameters():
    #    print(f"{name}: requires_grad={param.requires_grad}")
    #print(net.parameters())
    optimizer, loss_fct = init_optimization(net, args)
    # for param_group in optimizer.param_groups:
    #     for param in param_group['params']:
    #         print(param.shape, param.requires_grad)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume
    if args.RESUME_FROM > 0:
        ckpt = torch.load(os.path.join(opt.MODEL_DIR, '{}_{}_{}.pth'.format(args.model_name, opt.DATASET, args.RESUME_FROM)))
        net.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt.keys():
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt.keys():
            scheduler.load_state_dict(ckpt['scheduler'])
        epoch_start = args.RESUME_FROM

    net.cuda()

    # Training
    best_metric = -10000
    for epoch in range(epoch_start, epochs):
        net.train()
        print_str = '-------epoch {}/{}-------'.format(epoch+1, epochs)
        logger.write_and_print(print_str)

        for step, sample in enumerate(tqdm(train_loader)):

            patch = sample['image'].cuda()
            mask = sample['label']

            # prepare data
            batches_done = len(train_loader) * epoch + step
            optimizer.zero_grad()

            ann_array = []
            labels = []
            #print(patch.shape, mask.shape)
            #torch.Size([12, 3, 128, 128]) torch.Size([12, 4, 128, 128])
            for i in range(patch.shape[0]):
                random_idx = args.annotater #np.random.randint(0, args.mask_num)
                #print(random_idx)
                ann_array.append(random_idx)
                labels.append(mask[i, random_idx].unsqueeze(0).unsqueeze(0))
            labels = torch.cat(labels, dim=0).cuda()
            ann_array = torch.Tensor(ann_array).cuda().float()

            loss, _ = net.train_step(patch, labels, loss_fct, ann_array)
            if torch.isnan(loss):
                logger.write_and_print('***** Warning: loss is NaN *****')
                loss = torch.tensor(10000).cuda()

            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss', loss.item(), batches_done)

        # log learning_rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        scheduler.step()

        # save model
        # if epoch % 20 == 0:
        #     ckpt = {'model': net.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict()
        #             }
        #     torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}.pth'.format(args.model_name, opt.DATASET, epoch)))

        # validate each epoch
        metrics_dict = validate(net, val_loader, opt, writer, epoch, all_metrics=True, annotater=args.annotater)
        print_str = ''
        for key in metrics_dict.keys():
            print_str += key + ': {:.4f}  '.format(metrics_dict[key])
            writer.add_scalar('Metrics/'+key, metrics_dict[key], epoch)
        logger.write_and_print(print_str)
        #metrics_dict['GED'] + 
        metric_instance_ = metrics_dict['Dice_soft'] #- metrics_dict['GED']
        if metric_instance_ >= best_metric:
            best_metric = metric_instance_
            logger.write_and_print("Best Dice Max and Mean: {}".format(best_metric))
            ckpt = {'model': net.state_dict()}
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_best.pth'.format(args.model_name, opt.DATASET, epoch)))
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_best.pth'.format(args.model_name, opt.DATASET)))
    eval_dir = "/data/D-Persona/"+opt.MODEL_DIR[3:]
    import subprocess
    command = f"python evaluate_npc_unet.py --model_name unet --save_path {eval_dir}"
    # 执行命令
    subprocess.run(command, shell=True, check=True)
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
