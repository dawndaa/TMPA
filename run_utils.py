import torch
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

import json
import os

IMAGENET_VARIANTS = ['A', 'R', 'K', 'V', 'I']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):   
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)  
    topk = int(batch_entropy.size()[0] * top) if isinstance(top, float) else top
    idx = torch.argsort(batch_entropy, descending=False)[:topk]
    return logits[idx], idx

def select_confident_samples_seg(prob_maps: torch.Tensor, top: float):

    B_img, B_txt, C, H, W = prob_maps.shape
    entropy_map = -(prob_maps * torch.log(prob_maps.clamp(min=1e-8))).sum(dim=2)  
    image_entropy = entropy_map.mean(dim=(2, 3))  
    image_entropy = image_entropy.mean(dim=1)  
    topk = int(B_img * top) if isinstance(top, float) else top
    idx = torch.argsort(image_entropy, descending=False)[:topk]   

    return prob_maps[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) 
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) 
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def loss_prompt_logit(logits, logits_post, logit_scale, args): 

    probs = []
    for i in range(len(logits)):
        seg_probs = logits[i] * logit_scale   #[cls_num, w_ori, h_ori]
        seg_probs = F.softmax(seg_probs, dim=0)
        probs.append(seg_probs)
        
    mean_prob = logits_post.detach()
    loss = 0.0
    for p in probs:
        loss += F.mse_loss(p, mean_prob)
    return loss

def loss_prompt_entropy(logits, logits_post, logit_scale, args): 

    loss = 0.0
    for i in range(len(logits)):
        seg_probs = logits[i] * logit_scale   #[cls_num, w_ori, h_ori]
        seg_probs = F.softmax(seg_probs, dim=0)
        loss_ = avg_entropy_seg (seg_probs, args.prompt_logit_scale)
        loss += loss_
        # print(f'loss {i}: {loss_}')
    loss = loss / len(logits)

    return loss


def avg_entropy_seg(prob_maps: torch.Tensor, scale) -> torch.Tensor:

    prob_maps = prob_maps.clamp(min=1e-8)  
    entropy = -(prob_maps * torch.log(prob_maps)).sum(dim=0)  

    return entropy.mean() 

def log_results(top1, top5, batch_time, logname, set_id, tta_steps, bs, lr, concept_type=None, seed=None):
    os.makedirs('results', exist_ok=True)

    logpath = os.path.join('results', f"{logname}.json")

    results = {'dataset': set_id, 'concept_type': concept_type, 'batch_size': bs, 'lr:': lr, 'seed': seed, 'tta_steps': tta_steps, 'top1_avg': top1.item(), 'top5_avg': top5.item(), 'batch_time_avg': batch_time}
    with open(logpath, 'a') as f:
        f.write('\n')
        json.dump(results, f)

def compute_avg_cosine_sim(img_embeds, text_embeds):
    # both inputs are (bs, d)
    cosine_sims = F.normalize(img_embeds, dim=-1) * F.normalize(text_embeds, dim=-1)
    cosine_sims = torch.sum(cosine_sims, dim=-1)
    return cosine_sims.mean().item()