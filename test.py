import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import csv
import requests
import argparse
import lpips

from utils import util_calculate_psnr_ssim as util_psnr
from utils import utils_option as option
import torch.nn as nn

_result_psnr = list()
_result_ssim = list()
_result_psnr_y = list()
_result_ssim_y = list()
_result_lpips = list()

def main(opt, ema, n_model=500000, benchmark='Set5'):
    global _result_psnr, _result_ssim, _result_psnr_y, _result_ssim_y, _result_lpips

    if ema:
        model_path = f'superresolution/{opt["task"]}/models/{n_model}_E.pth'
    else:
        model_path = f'superresolution/{opt["task"]}/models/{n_model}_G.pth'
    opt['model_path'] = model_path
    opt['benchmark'] = benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
    else:
        raise FileNotFoundError(f'{model_path} file not found')

    model = define_model(opt)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder = f'testsets/{benchmark}/HR'
    save_dir = f'results/{opt["task"]}/{benchmark}/{n_model}'
    border = opt['scale']

    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['lpips'] = []

    psnr, ssim, psnr_y, ssim_y, lpips_score = 0, 0, 0, 0, 0
    
    loss_fn = lpips.LPIPS(net='alex').to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(opt, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to BCHW-RGB

        # inference
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()

            output = forward(img_lq, model, opt)
            output = output[..., :h_old * opt['scale'], :w_old * opt['scale']]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            # CHW-RGB to HWC-BGR
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        if ema:
            cv2.imwrite(f'{save_dir}/{imgname}_{opt["task"]}_tile{opt["tile"]}_E.png', output)
        else:
            cv2.imwrite(f'{save_dir}/{imgname}_{opt["task"]}_tile{opt["tile"]}_G.png', output)

        # evaluate Lpips
        
        gt_lpips = lpips.im2tensor(lpips.load_image(path)).to(device) # RGB image from [-1,1]
        if ema:
            sr_lpips = lpips.im2tensor(lpips.load_image(f'{save_dir}/{imgname}_{opt["task"]}_tile{opt["tile"]}_E.png')).to(device)
        else:
            sr_lpips = lpips.im2tensor(lpips.load_image(f'{save_dir}/{imgname}_{opt["task"]}_tile{opt["tile"]}_G.png')).to(device)

        lpips_score = loss_fn.forward(gt_lpips, sr_lpips)
        lpips_score = '%.4f' %lpips_score

        test_results['lpips'].append(float(lpips_score))

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            # float32 to uint8
            img_gt = (img_gt * 255.0).round().astype(np.uint8)
            img_gt = img_gt[:h_old * opt['scale'],:w_old * opt['scale'], ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util_psnr.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util_psnr.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # BGR image
                psnr_y = util_psnr.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util_psnr.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim/lpips
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])

        _result_psnr.append(ave_psnr)
        _result_psnr_y.append(ave_psnr_y)
        _result_ssim.append(ave_ssim)
        _result_ssim_y.append(ave_ssim_y)
        _result_lpips.append(ave_lpips)
        print('{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / \
                len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / \
                len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM(RGB)_Y/LPIPS: {:.2f} dB; {:.4f}; {:.4f}'.format(ave_psnr_y, ave_ssim_y, ave_lpips))
            print()


def define_model(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    
    if net_type == 'swinir_nat':
        from networks.uniwin import Uniwin as net
        model = net(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            img_size=opt_net['img_size'],
            window_size=opt_net['window_size'],
            kernel_size=opt_net['kernel_size'],
            img_range=opt_net['img_range'],
            depths=opt_net['depths'],
            embed_dim=opt_net['embed_dim'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            upsampler=opt_net['upsampler'],
            resi_connection=opt_net['resi_connection'],
            layer_scale=opt_net['layer_scale']
        )

        param_key_g = 'params'
        pretrained_model = torch.load(opt['model_path'])
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    else:
        raise NotImplementedError(f'{net_type} not implemented')

    return model


def get_image_pair(opt, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(
        np.float32) / 255.  # HWC-BGR [0,1]
    img_lq = cv2.imread(f'testsets/{opt["benchmark"]}/LR_bicubic/X{opt["scale"]}/{imgname}x{opt["scale"]}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def forward(img_lq, model, opt):

    if opt['tile'] is not None:
        # test the image tile by tile
        tile = opt['tile']

        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile//2
        assert tile % opt['netG']['window_size'] == 0, "tile size should be a multiple of window_size"

        sf = opt['scale']

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)[0]
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx *
                  sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx *
                  sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)
    else:
        # test the image as a whole
        output = model(img_lq)

    return output[0]


def test(args):
    global _result_psnr, _result_ssim, _result_psnr_y, _result_ssim_y, _result_lpips

    opt = option.parse(args.opt)
    n_iter = int(args.max_iter/args.unit_iter)

    opt['tile'] = args.tile

    if not isinstance(args.benchmarks, list):
        args.benchmarks = [args.benchmarks]

    for benchmark in args.benchmarks:
        print_benchmark = '#' * 20 + f' TEST BENCHMARK [{benchmark}] ' + '#' * 20
        print_sharp = '#'*len(print_benchmark)

        print(print_sharp)
        print(print_benchmark)
        print(print_sharp)
        print()
        _iter_list = list()
        for n in range(n_iter):
            _iter = (n+1)*args.unit_iter
            _iter_list.append(_iter)

            main(opt, args.ema, n_model=_iter, benchmark=benchmark)

        res = list()
        res.append(_iter_list)                                                                                                                                                                                                                                                                                                      
        res.append(_result_psnr)
        res.append(_result_ssim)
        res.append(_result_psnr_y)
        res.append(_result_ssim_y)
        res.append(_result_lpips)

        data_to_write = zip(*res)
        save_path = f'benchmark-results/{opt["task"]}/{_iter}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if '/' in benchmark:
            csv_name = '_'.join(benchmark.split('/'))
        else:
            csv_name = benchmark

        if args.ema:
            save_name = f'{csv_name}_result_E_tile{opt["tile"]}.csv'
        else:
            save_name = f'{csv_name}_result_G_tile{opt["tile"]}.csv'

        with open(save_path + save_name, 'w', newline='') as fw:
            wr = csv.writer(fw)
            wr.writerow(('iter', 'psnr', 'ssim', 'psnr_y', 'ssim_y', 'lpips'))
            for data in data_to_write:
                wr.writerow(data)
        _result_psnr = list()
        _result_psnr_y = list()
        _result_ssim = list()
        _result_ssim_y = list()
        _result_lpips = list()


if __name__ == '__main__':
    test_opt_list = ['uniwin_x2_patch64_win16_kernel9_Imagenet800K']
    test_benchmark_list = ['Set5', 'Set14', 'BSDS100', 'urban100', 'manga109']

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--tile', type=int, default=None)
    parser.add_argument('--benchmarks', type=str,
                        default=test_benchmark_list)
    parser.add_argument('--unit_iter', type=int, default=800000)
    parser.add_argument('--max_iter', type=int, default=800000)
    args = parser.parse_args()

    if args.opt is not None:
        test(args)
    else:
        for test_opt in test_opt_list:
            args.opt = f'options/{test_opt}.json'
            test(args)
