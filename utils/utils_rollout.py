import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2


def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)

def rollout(attentions, discard_ratio, head_fusion, w, h):
    print(f'###')
    print(f'### type(attentions): {type(attentions)}')
    print(f'### len(attentions): {len(attentions)}')
    print(f'### attentions[0].shape: {attentions[0].shape}')
    print(f'###')

    len_attentions = len(attentions)

    attentions_hb = attentions[:(len_attentions//2)]
    attentions_sb = attentions[(len_attentions//2):]

    print(f'###')
    print(f'### len(attentions_hb): {len(attentions_hb)}')
    print(f'### len(attentions_sb): {len(attentions_sb)}')
    print(f'###')

    result_hb = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions_hb:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            flat = attention_heads_fused.view(1, -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            flat[0, indices] = 0
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1).unsqueeze(dim=-1)

            result_hb = torch.matmul(a, result_hb)

    mask_hb = result_hb.mean(axis=1)
    width = int(mask_hb.size(-1)**0.5)
    mask_hb = mask_hb.reshape(-1, width, width).numpy()

    n_width = w // width
    n_height = h // width

    mask_hb = mask_hb.reshape(n_width, n_height, width, width)
    mask_hb = np.transpose(mask_hb, (0, 2, 1, 3))
    mask_hb = mask_hb.reshape(w, h)

    # n_width = int(mask_hb.shape[0]**0.5)
    # mask_hb = mask_hb[:n_width*n_width, :, :]
    # mask_hb = mask_hb.reshape(n_width, n_width, width, width)
    # mask_hb = np.transpose(mask_hb, (0, 2, 1, 3))
    # mask_hb = mask_hb.reshape(mask_hb.shape[0]*mask_hb.shape[1], -1)
    mask_hb = mask_hb / np.max(mask_hb)

    result_sb = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions_sb:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            # print('##########################')
            # print(attention.shape)
            # print(attention_heads_fused.shape)
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(1, -1)
            # flat = attention_heads_fused.view(
            #     attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            # indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1).unsqueeze(dim=-1)

            result_sb = torch.matmul(a, result_sb)

    mask_sb = result_sb.mean(axis=1)
    width = int(mask_sb.size(-1)**0.5)
    mask_sb = mask_sb.reshape(-1, width, width).numpy()

    n_width = w // width
    n_height = h // width

    mask_sb = mask_sb.reshape(n_width, n_height, width, width)
    mask_sb = np.transpose(mask_sb, (0, 2, 1, 3))
    mask_sb = mask_sb.reshape(w, h)

    # n_width = int(mask_sb.shape[0]**0.5)
    # mask_sb = mask_sb[:n_width*n_width, :, :]
    # mask_sb = mask_sb.reshape(n_width, n_width, width, width)
    # mask_sb = np.transpose(mask_sb, (0, 2, 1, 3))
    # mask_sb = mask_sb.reshape(mask_sb.shape[0]*mask_sb.shape[1], -1)
    mask_sb = mask_sb / np.max(mask_sb)

    # print(f'###')
    # print(f'### mask_hb.shape: {mask_hb.shape}')
    # print(f'### mask_sb.shape: {mask_sb.shape}')
    # print(f'###')

    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, 0, 1:]
    # mask_sb = result_sb.mean(axis=1)
    # mask = result[:, -1, :]

    # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1)**0.5)
    # mask = mask.reshape(-1, width, width).numpy()
    # mask = mask / np.max(mask)

    return mask_hb, mask_sb


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion, input_tensor.shape[2], input_tensor.shape[3])
