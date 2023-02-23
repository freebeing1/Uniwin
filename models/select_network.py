import functools
import torch
from torch.nn import init
from torchinfo import summary
"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""

# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    # ----------------------------------------
    # MySR 
    # ----------------------------------------
    if net_type == 'dense2sparse':
        from networks.network_swinir_dense2sparse import SwinIR as net
        netG = net(
            img_size=opt_net['img_size'],
            in_chans=opt_net['in_chans'],
            embed_dim=opt_net['embed_dim'],
            depths = opt_net['depths'],
            num_heads=opt_net['num_heads'],
            window_size=opt_net['window_size'],
            mlp_ratio=opt_net['mlp_ratio'],
            upscale=opt_net['upscale'],
            img_range=opt_net['img_range'],
            upsampler=opt_net['upsampler'],
            resi_connection = opt_net['resi_connection'],
            global_partition= opt_net['global_partition']
        )
    elif net_type == 'dense2sparse_srpb':
        from networks.network_swinir_dense2sparse_srpb import SwinIR as net
        netG = net(
            img_size=opt_net['img_size'],
            in_chans=opt_net['in_chans'],
            embed_dim=opt_net['embed_dim'],
            depths = opt_net['depths'],
            num_heads=opt_net['num_heads'],
            window_size=opt_net['window_size'],
            mlp_ratio=opt_net['mlp_ratio'],
            upscale=opt_net['upscale'],
            img_range=opt_net['img_range'],
            upsampler=opt_net['upsampler'],
            resi_connection = opt_net['resi_connection'],
            global_partition= opt_net['global_partition']
        )
    elif net_type == 'dense2sparse_no_pos':
        from networks.network_swinir_dense2sparse_no_pos import SwinIR as net
        netG = net(
            img_size=opt_net['img_size'],
            in_chans=opt_net['in_chans'],
            embed_dim=opt_net['embed_dim'],
            depths = opt_net['depths'],
            num_heads=opt_net['num_heads'],
            window_size=opt_net['window_size'],
            mlp_ratio=opt_net['mlp_ratio'],
            upscale=opt_net['upscale'],
            img_range=opt_net['img_range'],
            upsampler=opt_net['upsampler'],
            resi_connection = opt_net['resi_connection'],
            global_partition= opt_net['global_partition']
        )
    elif net_type == 'swinir':
        from networks.network_swinir import SwinIR as net
        netG = net(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            img_size=opt_net['img_size'],
            window_size=opt_net['window_size'],
            img_range=opt_net['img_range'],
            depths=opt_net['depths'],
            embed_dim=opt_net['embed_dim'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            upsampler=opt_net['upsampler'],
            resi_connection=opt_net['resi_connection']
        )
    elif net_type == 'dense2sparse_attnmask':
        from networks.network_dense2sparse_attnmask import SwinIR as net
        netG = net(
            img_size=opt_net['img_size'],
            in_chans=opt_net['in_chans'],
            embed_dim=opt_net['embed_dim'],
            depths=opt_net['depths'],
            num_heads=opt_net['num_heads'],
            window_size=opt_net['window_size'],
            mlp_ratio=opt_net['mlp_ratio'],
            upscale=opt_net['upscale'],
            img_range=opt_net['img_range'],
            upsampler=opt_net['upsampler'],
            resi_connection=opt_net['resi_connection'],
            global_partition=opt_net['global_partition']
        )
    elif net_type == 'natsr':
        from networks.network_nat import NATSR as net
        netG = net(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            embed_dim=opt_net['embed_dim'],
            kernel_size=opt_net['kernel_size'],
            img_range=opt_net['img_range'],
            depths=opt_net['depths'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            resi_connection=opt_net['resi_connection'],
            layer_scale=opt_net['layer_scale'],
            dilations=None
        )
    elif net_type == 'dinatsr':
        from networks.network_nat import NATSR as net
        netG = net(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            embed_dim=opt_net['embed_dim'],
            kernel_size=opt_net['kernel_size'],
            img_range=opt_net['img_range'],
            depths=opt_net['depths'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            resi_connection=opt_net['resi_connection'],
            layer_scale=opt_net['layer_scale'],
            dilations=opt_net['dilations']
        )
    elif net_type == 'uniwin':
        from networks.network_uniwin import Uniwin as net
        netG = net(
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
    elif net_type == 'swinir_nat_2':
        from networks.network_swinir_nat_2 import SwinIR_NAT as net
        netG = net(
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
    elif net_type == 'd2s_nat':
        from networks.network_d2s_nat import D2S_NAT as net
        netG = net(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            img_size=opt_net['img_size'],
            window_size=opt_net['window_size'],
            kernel_size=opt_net['kernel_size'],
            img_range=opt_net['img_range'],
            depths=opt_net['depths'],
            global_partition=opt_net['global_partition'],
            embed_dim=opt_net['embed_dim'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            upsampler=opt_net['upsampler'],
            resi_connection=opt_net['resi_connection'],
            layer_scale=opt_net['layer_scale']
        )
    else:
        raise NotImplementedError(f'net type [{net_type}] not implemented')
        # netG = net(upscale=opt_net['upscale'],
        #             in_chans=opt_net['in_chans'],
        #             img_range=opt_net['img_range'],
        #             img_size=opt_net['img_size'],
        #             window_size=opt_net['window_size'],
        #             token_size=opt_net['token_size'],
        #             num_feat=opt_net['num_feat'],
        #             num_layers=opt_net['num_layers'],
        #             num_heads=opt_net['num_heads'],
        #             num_resblocks=opt_net['num_resblocks'],
        #             expansion_ratio=opt_net['expansion_ratio'],
        #             mlp_ratio=opt_net['mlp_ratio'])
    # if opt['rank'] == 0:
    #     summary(netG, depth =5)
    
    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0,
                                     mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0,
                                      mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError(
                    'Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    'Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print(
            'Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type,
                               init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print(
            'Pass this initialization! Initialization was done during network definition!')
