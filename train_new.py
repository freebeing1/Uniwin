import os
import logging
import random
import numpy as np
import math
import cv2


import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import build_dataset
from utils import utils_logger
from utils import utils_image as util
from utils import util_calculate_psnr_ssim as util_calculate
from utils import utils_option as option
from models.model_sr import ModelSR as define_Model
from torchinfo import summary


class Trainer:
    def __init__(
        self,
        opt: str,
    ) -> None:

        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])

        self.opt = self.init_opt(opt)
        self.init_logger()
        self.init_seed()

        self.prepare_dataloader()

        self.model = define_Model(self.opt)
        self.model.init_train()

        self.print_arch()



    def init_opt(self, opt):

        opt = option.parse(opt, is_train=True)

        self.init_checkpoint_path(opt)
        self.current_step = self.update_checkpoint(opt)

        # return None for missing key
        opt = option.dict_to_nonedict(opt)

        return opt


    def init_checkpoint_path(self, opt):

        if self.global_rank == 0:
            util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))


    def update_checkpoint(self, opt):

        init_iter_G, init_path_G = option.find_last_checkpoint(
            opt['path']['models'], net_type='G', pretrained_path=opt['path']['pretrained_netG'])
        init_iter_E, init_path_E = option.find_last_checkpoint(
            opt['path']['models'], net_type='E', pretrained_path=opt['path']['pretrained_netE'])        
        opt['path']['pretrained_netG'] = init_path_G
        opt['path']['pretrained_netE'] = init_path_E
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
            opt['path']['models'], net_type='optimizerG')    
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG

        # save opt to  a '../option.json' file
        if self.global_rank == 0:
            option.save(opt)

        return max(init_iter_G, init_iter_E, init_iter_optimizerG)


    def init_logger(self, logger_name='train'):
        if self.local_rank == 0:
            utils_logger.logger_info(logger_name, os.path.join(
                self.opt['path']['log'], logger_name+'.log'))
            self.logger = logging.getLogger(logger_name)
            self.logger.info(option.dict2str(self.opt))


    def init_seed(self):

        seed = self.opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        if self.opt['global_rank'] == 0:
            self.logger.info(f'Random seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed
        
        

    def prepare_dataloader(self):

        for phase, dataset_opt in self.opt['datasets'].items():
            
            if phase == 'train':
                
                train_set = build_dataset(dataset_opt)
                self.n_train_images = len(train_set)

                self.train_size = int(math.ceil(self.n_train_images / dataset_opt['dataloader_batch_size']))
                
                if self.global_rank == 0:
                    self.logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                        self.n_train_images, self.train_size))
                
                self.train_sampler = DistributedSampler(
                    train_set, 
                    shuffle=dataset_opt['dataloader_shuffle'], 
                    drop_last=True, 
                    seed=self.seed)

                self.train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt['dataloader_batch_size']//self.opt['num_gpu'],
                    shuffle=False,
                    num_workers=dataset_opt['dataloader_num_workers']//self.opt['num_gpu'],
                    drop_last=True,
                    pin_memory=True,
                    sampler=self.train_sampler)

            elif phase == 'test':
                
                test_set = build_dataset(dataset_opt)
                
                self.test_loader = DataLoader(
                    test_set, 
                    batch_size=1,
                    shuffle=False, 
                    num_workers=1,
                    drop_last=False, 
                    pin_memory=True)

            else:
                raise NotImplementedError("Phase [%s] is not recognized." % phase)


    def print_arch(self, depth=5):
        if self.local_rank == 0:
            summary(self.model.netG, depth=depth)


    def train(self):
        
        epochs_run = self.current_step // self.train_size

        for epoch in range(epochs_run, epochs_run + 1000000):
            
            self.train_sampler.set_epoch(epoch)

            for i, train_data in enumerate(self.train_loader):
                self.current_step += 1

                self.model.update_learning_rate(self.current_step)
                self.model.feed_data(train_data)
                self.model.optimize_parameters(self.current_step)

                # print training information
                if self.current_step % self.opt['train']['checkpoint_print'] == 0 and self.local_rank == 0:
                    logs = self.model.current_log()  # such as loss
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                        epoch, self.current_step, self.model.current_learning_rate())
                    for k, v in logs.items():  # merge log information into message
                        message += '{:s}: {:.3e} '.format(k, v)
                    self.logger.info(message)

                # save model
                if self.current_step % self.opt['train']['checkpoint_save'] == 0 and self.local_rank == 0:
                    self.logger.info('Saving the model.')
                    self.model.save(self.current_step)

                # testing
                if self.current_step % self.opt['train']['checkpoint_test'] == 0 and self.global_rank == 0:

                    avg_psnr = 0.0
                    avg_psnr_y = 0.0
                    idx = 0

                    for test_data in self.test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)

                        img_dir = os.path.join(self.opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                        self.model.feed_data(test_data)
                        self.model.test()

                        visuals = self.model.current_visuals(need_HF=True)
                        SR_img = util.tensor2uint(visuals['SR'])  # HWC-RGB
                        H_img = util.tensor2uint(visuals['H'])  # HWC-RGB

                        # -----------------------
                        # save estimated image SR
                        # -----------------------
                        SR_save_img_path = os.path.join(
                            img_dir, f'{img_name}_{self.current_step}.png')
                        util.imsave(SR_img, SR_save_img_path)

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        current_psnr = util_calculate.calculate_psnr(
                            SR_img, 
                            H_img, 
                            crop_border=self.opt['scale'])
                        current_psnr_y = util_calculate.calculate_psnr(
                            cv2.cvtColor(SR_img, cv2.COLOR_BGR2RGB), 
                            cv2.cvtColor(H_img, cv2.COLOR_BGR2RGB), 
                            crop_border=self.opt['scale'], 
                            test_y_channel=True)
                        self.logger.info(
                            '{:->4d}--> {:>10s} | {:<4.2f}dB | (PSNR_Y {:<4.2f}dB)'.format(
                                idx, image_name_ext, current_psnr, current_psnr_y))

                        avg_psnr += current_psnr
                        avg_psnr_y += current_psnr_y

                    avg_psnr = avg_psnr / idx
                    avg_psnr_y = avg_psnr_y / idx

                    # testing log
                    self.logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average PSNR_Y : {:<.2f}dB'.format(
                        epoch, self.current_step, avg_psnr, avg_psnr_y))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Training configuraion json file path')

    
    # DDP(DistributedDataParallel) setup
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(int(os.environ['RANK']) % world_size)
    init_process_group(backend='nccl')

    trainer = Trainer(parser.parse_args().opt)
    trainer.train()
    
    destroy_process_group()