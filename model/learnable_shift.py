import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from model import load, DOWNLOAD_ROOT
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

from data.prompt_embeds import get_class2concept_dict_path, get_concept_embeds_path, get_class_embeds_path, get_susx_class_embeds_path

from .text_encoders import ClipTextEncoder
from .visual_encoders import ClipImageEncoder

from .shifter import Shifter_seg,Shifter, Shifter_visual
from .film import FiLM

import pywt

from prompts.imagenet_template import *

import numpy as np   
import matplotlib.pyplot as plt  
import os   
import ot

import re

from open_clip import tokenizer, create_model 
from simfeatup_dev.upsamplers import get_upsampler  
from torchvision.utils import save_image 
from visualize import visualize_and_save_feature
from model.dataset_config import DATASET_CONFIGS

def get_dataset_config(dataset_name, resolution=None):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_cfg = DATASET_CONFIGS[dataset_name]

    if resolution is not None:
        print("resolution:", resolution)
        resolution = str(resolution)

        if resolution in dataset_cfg:
            return dataset_cfg[resolution]

    raise ValueError(
        f"No config found for dataset={dataset_name}, "
        f"resolution={resolution}"
    )

class TestTimeShiftTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, arch="ViT-B/16", 
                        test_set=None,
                        # concept_type='labo', 
                        init_concepts=False,
                        per_label=False,
                        with_concepts=False,
                        text_shift=False,
                        do_shift=True,
                        do_scale=False,
                        do_film=False,
                        with_templates=False,
                        macro_pooling=False,
                        with_coop=False,
                        use_susx_feats=False,
                        args = None
                    ):
        super(TestTimeShiftTuning, self).__init__()

        self.arch = arch
        self.device = device
        self.batch_size = batch_size

        config = get_dataset_config(
            args.dataset_name,
            args.resolution
        )   
        self.prob_thd = config["prob_thd"]    
        self.cls_token_lambda = config["cls_token_lambda"]   
        self.logit_scale = config["logit_scale"]  
        self.logit_weight = config["logit_weight"]  
        self.bg_idx = config["bg_idx"]   
        self.alpha_init = config["alpha"]  

        self.classnames = classnames
        self.init_concepts = init_concepts
        self.with_concepts = with_concepts

        self.per_label = per_label

        self.with_templates = with_templates
        self.with_coop = with_coop

        self.test_set = test_set
        self.use_susx_feats = use_susx_feats
        
        self.net = create_model('ViT-B/16', pretrained='openai', cache_dir=DOWNLOAD_ROOT)
        self.embed_dim = self.net.text_projection.shape[1]

        self.net.eval().to(device) 
        self.tokenizer = tokenizer.tokenize 

        self.output_cls_token = self.cls_token_lambda != 0   
        self.patch_size = self.net.visual.patch_size  
        self.ignore_residual = True   
        self.feature_up = True   

        query_words, query_idx = get_cls_idx_multi_prom(args.name_path)
        # Keep a CPU copy for prompt grouping. query_idx maps every description
        # prompt to its semantic class and is constant throughout evaluation.
        self.query_idx_list = [int(idx) for idx in query_idx]
        self.query_idx = torch.tensor(self.query_idx_list, dtype=torch.int64, device=device)
        query_features = []
        with torch.no_grad(): 
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)
        self.text_embeds = self.query_features

        # v4 separates prediction prompts from the frozen multi-description
        # reliability bank. When Cat-Prompt is enabled, both are identical so
        # the Full RSAP+SDR path remains behaviorally unchanged. When disabled,
        # prediction uses one prompt per class while SDR can still estimate
        # reliability from the preserved original multi-description prompt file.
        if args.module_cat_prompt:
            self.reliability_query_idx_list = list(self.query_idx_list)
            self.reliability_query_features = self.query_features.detach()
        elif args.loss_sdr:
            reliability_words, reliability_idx = get_cls_idx_multi_prom(
                args.reliability_prompt_path
            )
            self.reliability_query_idx_list = [int(idx) for idx in reliability_idx]
            reliability_features = []
            with torch.no_grad():
                for qw in reliability_words:
                    query = self.tokenizer(
                        [temp(qw) for temp in openai_imagenet_template]
                    ).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    reliability_features.append(feature.unsqueeze(0))
            self.reliability_query_features = torch.cat(
                reliability_features, dim=0
            ).detach()
        else:
            # No reliability consumer is active in this configuration.
            self.reliability_query_idx_list = []
            self.reliability_query_features = None

        num_queries = len(self.query_idx)   
        self.alpha = nn.Parameter(torch.full((num_queries, 1), self.alpha_init, device=device, dtype=self.dtype))    

        self.last_reliability_map = None
        if self.feature_up:   
            self.feat_dim = self.embed_dim 
            self.upsampler = get_upsampler('jbu_one', self.feat_dim).cuda().half()
            ckpt = torch.load('simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt')['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)
            
            for param in self.upsampler.parameters():     
                param.requires_grad = False

        self.class_embeds = None

        self.text_shift = text_shift   
        self.do_shift = do_shift   
        self.do_scale = do_scale   
        self.do_film = do_film    

        self.num_classes = len(classnames)  
        self.num_queries = self.num_classes 

        if self.text_shift:    
            self.text_shifter = self.load_shifter_seg(text_embeds=self.query_features, per_label=self.per_label)
            self.text_shifter_visual = self.load_shifter_visual(text_embeds=self.text_embeds, per_label=self.per_label)
    
    @property
    def dtype(self):
        return self.net.visual.conv1.weight.dtype
    
    def load_shifter(self, embed_dim, per_label=False):
        num = self.num_classes if (not self.init_concepts) else self.num_concepts   #7
        
        if self.do_film:   
            return FiLM(
                embed_dim,
                dtype=self.dtype, 
                num_classes=num,
                per_label=per_label,
                text_embeds=self.text_embeds,
                device=self.device
            )
        else:
            return Shifter(
                embed_dim,   
                dtype=self.dtype, 
                do_shift=self.do_shift,   
                do_scale=self.do_scale,    
                num_classes=num,        
                per_label=per_label,   
                text_embeds=self.text_embeds,
                class2concepts=self.class2concepts,  
                device=self.device
            )
        
    def load_shifter_seg(self, text_embeds, per_label=False):
        num = self.num_classes if (not self.init_concepts) else self.num_concepts  
        
        if self.do_film:  
            return FiLM(
                embed_dim=None,
                dtype=self.dtype, 
                num_classes=num,
                per_label=per_label,
                text_embeds=self.text_embeds,
                device=self.device
            )
        else:
            return Shifter_seg(
                dtype=self.dtype, 
                do_shift=self.do_shift, 
                do_scale=self.do_scale,   
                num_classes=num,      
                per_label=per_label,  
                text_embeds=text_embeds,   
                class2concepts=None,   
                device=self.device
            )
        
    def load_shifter_visual(self, text_embeds, per_label=False):
        num = self.num_classes if (not self.init_concepts) else self.num_concepts   
        
        if self.do_film: 
            return FiLM(
                embed_dim=None,
                dtype=self.dtype, 
                num_classes=num,
                per_label=per_label,
                text_embeds=self.text_embeds,
                device=self.device
            )
        else:
            return Shifter_visual(
                dtype=self.dtype, 
                do_shift=self.do_shift,  
                do_scale=self.do_scale,  
                num_classes=num,    
                per_label=per_label,  
                text_embeds=text_embeds,
                class2concepts=None, 
                device=self.device
            )

    def reset(self, args):
        if self.text_shift:
            self.text_shifter.reset()
            self.text_shifter_visual.reset()

        with torch.no_grad():
            self.alpha.fill_(args.alpha)

    def get_img_features(self, img):
        img = img.type(self.dtype)
        with torch.no_grad():
            img_features = self.net.encode_image(img)

        img_features = F.normalize(img_features, dim=-1)

        return img_features

    def get_text_features(self, test=False):
        text_features = self.text_embeds  

        
        if isinstance(text_features, torch.Tensor):
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = [F.normalize(t, dim=-1) for t in text_features]

        return text_features   

    def _average_concept_embeds(self, text_features):
        num_concepts_per_class = [len(self.class2concepts[c]) for c in self.classnames]

        new_text_features = torch.empty((self.num_classes, text_features.shape[1]), dtype=self.dtype, device=self.device)

        i = 0
        for j, n in enumerate(num_concepts_per_class):
            new_text_features[j] = text_features[i:i+n].mean(dim=0)
            i += n
        
        return new_text_features

    def _rsap_consensus_visual_mining(
        self,
        image_features_clip,
        prompt_features,
        prompt_idx_list,
        topk=3,
        gamma=1.0,
        use_rsap_reliability=True,
    ):
        """Build reliability-weighted visual prototypes from multi-prompt consensus.

        Prompt files can contain a different number of descriptions per class
        (LoveDA is one example), so RSAP does not align the k-th prompt across
        classes. Instead, for each class c and each of its prompts k, that prompt
        replaces c in an otherwise class-mean prototype bank and yields an
        independent C-way probability p_{i,c}^{(k)}. Consensus and disagreement
        are then computed across all prompts belonging to c.

        Reliability is detached from the optimization graph because it is an
        evidence-selection/gating statistic rather than another target that
        entropy minimization should be able to manipulate directly.

        use_rsap_reliability=False disables only RSAP ranking, prototype
        weighting and fusion gating. The returned token reliability remains
        the original estimate for SDR and visualization. Present classes use
        unit fusion gates; absent classes retain zero visual contribution.
        """
        if image_features_clip.ndim != 3 or image_features_clip.shape[0] != 1:
            raise ValueError(
                'RSAP-v1 currently expects one crop/image at a time, matching TMPA sliding inference.'
            )

        class_to_prompt_indices = [[] for _ in range(self.num_classes)]
        for prompt_idx, class_idx in enumerate(prompt_idx_list):
            if class_idx < 0 or class_idx >= self.num_classes:
                raise ValueError(
                    f'Prompt class index {class_idx} is outside [0, {self.num_classes - 1}].'
                )
            class_to_prompt_indices[class_idx].append(prompt_idx)

        prompt_counts = [len(indices) for indices in class_to_prompt_indices]
        if not prompt_counts or min(prompt_counts) < 2:
            raise ValueError(
                'Reliability estimation requires at least two prompts for every class; '
                f'got per-class counts {prompt_counts}.'
            )

        with torch.no_grad():
            visual_tokens = F.normalize(image_features_clip[0].detach(), dim=-1)
            fixed_prompts = F.normalize(prompt_features.detach(), dim=-1)

            # Stable class references for the competing classes. The currently
            # evaluated class is replaced by each of its prompts in turn.
            class_mean_prompts = []
            for prompt_indices in class_to_prompt_indices:
                class_mean = fixed_prompts[prompt_indices].mean(dim=0)
                class_mean_prompts.append(F.normalize(class_mean, dim=-1))
            class_mean_prompts = torch.stack(class_mean_prompts, dim=0)

            class_consensus = []
            class_disagreement = []
            for class_idx, prompt_indices in enumerate(class_to_prompt_indices):
                per_prompt_support = []
                for prompt_idx in prompt_indices:
                    class_bank = class_mean_prompts.clone()
                    class_bank[class_idx] = fixed_prompts[prompt_idx]
                    view_logits = visual_tokens @ class_bank.T
                    view_probs = F.softmax(
                        view_logits.float() * float(self.logit_scale), dim=-1
                    )
                    per_prompt_support.append(view_probs[:, class_idx])

                prompt_support = torch.stack(per_prompt_support, dim=0)  # [K_c, N]
                class_consensus.append(prompt_support.mean(dim=0))
                class_disagreement.append(
                    prompt_support.var(dim=0, unbiased=False)
                )

            consensus = torch.stack(class_consensus, dim=-1)      # [N, C]
            disagreement = torch.stack(class_disagreement, dim=-1)
            reliability = consensus * torch.exp(-float(gamma) * disagreement)

            predicted_class = consensus.argmax(dim=-1)
            class_visual_prototypes = []
            class_reliability = []

            for class_idx in range(self.num_classes):
                class_mask = predicted_class == class_idx
                num_candidates = int(class_mask.sum().item())

                if num_candidates == 0:
                    prototype = torch.zeros(
                        visual_tokens.shape[-1],
                        device=visual_tokens.device,
                        dtype=visual_tokens.dtype,
                    )
                    reliability_score = reliability.new_zeros(())
                else:
                    k = min(int(topk), num_candidates)
                    if use_rsap_reliability:
                        scores = reliability[:, class_idx].masked_fill(
                            ~class_mask, torch.finfo(reliability.dtype).min
                        )
                        top_values, top_indices = torch.topk(scores, k=k, largest=True)
                        weights = top_values.clamp_min(1e-8)
                    else:
                        # Keep the same K-token budget without confidence ranking
                        # or consuming RNG state in online/reference forwards.
                        candidates = torch.where(class_mask)[0]
                        positions = (
                            (2 * torch.arange(k, device=candidates.device) + 1)
                            * num_candidates // (2 * k)
                        )
                        top_indices = candidates[positions]
                        weights = reliability.new_ones(k)
                    selected_tokens = visual_tokens[top_indices].float()
                    prototype = (
                        selected_tokens * weights.unsqueeze(-1)
                    ).sum(dim=0) / weights.sum()
                    prototype = prototype.to(visual_tokens.dtype)
                    reliability_score = weights.mean()

                class_visual_prototypes.append(prototype)
                class_reliability.append(reliability_score)

            class_visual_prototypes = torch.stack(class_visual_prototypes, dim=0)
            class_reliability = torch.stack(class_reliability, dim=0).clamp(0.0, 1.0)

            prompt_class_idx = self.query_idx.to(class_visual_prototypes.device)
            prompt_visual_prototypes = class_visual_prototypes[prompt_class_idx]
            prompt_reliability = class_reliability[prompt_class_idx].unsqueeze(-1)
            token_reliability = reliability.gather(
                1, predicted_class.unsqueeze(-1)
            ).squeeze(-1).clamp(0.0, 1.0)

        return prompt_visual_prototypes, prompt_reliability, token_reliability

    def forward_feature(self, img, image_name, args, logit_size=None):
        if type(img) == list:
            img = img[0]
        
        device = next(self.net.parameters()).device   
        img = img.to(device)    
        self.text_features = self.text_features.to(self.device)    

        self.output_cls_token = True
        reliability_map = None

        image_features = self.net.encode_image(img, self.ignore_residual, self.output_cls_token)  

        for param in self.net.visual.parameters():
            param.requires_grad = False

            
        if self.output_cls_token:
            image_cls_token, image_features = image_features  
            image_cls_token = image_cls_token / image_cls_token.norm(dim=-1, keepdim=True)

            cls_logits = image_cls_token.to(device) @ self.text_features.T.to(device)  

        needs_reliability = bool(args.module_rsap_v1 or args.loss_sdr)
        needs_prompt_adjustment = (
            args.text_adjust == 'True'
            and (args.module_rsap_v1 or args.module_visual_guidance)
        )
        if needs_reliability or needs_prompt_adjustment:
            image_features_clip = F.normalize(image_features.clone(), dim=-1)

            selected_features = None
            prompt_reliability = None
            if needs_reliability:
                (
                    selected_features,
                    prompt_reliability,
                    token_reliability,
                ) = self._rsap_consensus_visual_mining(
                    image_features_clip,
                    # The frozen reliability bank is independent from the
                    # segmentation-prediction prompt bank in v4. For Full
                    # RSAP+SDR they are identical; for SDR-only the former
                    # remains multi-description while prediction is single-prompt.
                    self.reliability_query_features.to(device),
                    self.reliability_query_idx_list,
                    topk=args.rsap_topk,
                    gamma=args.rsap_gamma,
                    use_rsap_reliability=not getattr(args, 'no_rsap_reliability', False),
                )
                feature_w = img[0].shape[-2] // self.patch_size[0]
                feature_h = img[0].shape[-1] // self.patch_size[1]
                expected_tokens = feature_w * feature_h
                if token_reliability.numel() != expected_tokens:
                    raise ValueError(
                        'Reliability token count does not match the visual grid: '
                        f'{token_reliability.numel()} vs {feature_w}x{feature_h}.'
                    )
                reliability_map = token_reliability.view(
                    1, 1, feature_w, feature_h
                )
                reliability_map = F.interpolate(
                    reliability_map.float(),
                    size=img.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).clamp(0.0, 1.0)

            if args.module_rsap_v1:
                # Default: gate calibration using reliability. The RSAP-only
                # ablation returns unit gates for present classes and zero for
                # absent classes, without changing SDR's token reliability.
                beta = (
                    self.alpha
                    * prompt_reliability.to(
                        device=self.alpha.device, dtype=self.alpha.dtype
                    )
                ).clamp(0.0, 1.0)
                updated_query_features = (
                    (1 - beta) * self.text_features.to(device)
                    + beta * selected_features.to(device)
                ).to(self.text_features.dtype)
                updated_query_features = F.normalize(
                    updated_query_features, dim=-1
                )
            elif needs_prompt_adjustment and args.module_visual_guidance:
                # Legacy TMPA Visual Guidance path. Standalone SDR does not enter
                # this branch when --no_module_visual_guidance is used.
                topk = 3
                logits = image_features_clip @ self.text_features.T.to(device)
                B, N, D = image_features_clip.shape
                C, Dq = self.text_features.shape
                assert D == Dq
                max_index_map = self.postprocess_entropy(
                    topk,
                    logits.permute(0, 2, 1).reshape(-1, C, 14, 14),
                    image_name,
                    img,
                )
                selected_features = []
                for i, idx in enumerate(self.query_idx):
                    max_indices = max_index_map[idx]
                    if max_indices is not None and len(max_indices) > 0:
                        feats = []
                        for max_idx in max_indices:
                            feat = image_features_clip[0, max_idx, :]
                            feats.append(feat)
                        feat = torch.stack(feats, dim=0).mean(dim=0)
                    else:
                        feat = torch.zeros(
                            image_features_clip.shape[2],
                            device=image_features_clip.device,
                            dtype=image_features_clip.dtype,
                        )
                    selected_features.append(feat)
                selected_features = torch.stack(selected_features, dim=0)
                updated_query_features = (
                    (1 - self.alpha) * self.text_features.to(device)
                    + self.alpha * selected_features.to(device)
                ).to(self.text_features.dtype)
                updated_query_features = F.normalize(
                    updated_query_features, dim=-1
                )


        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]   
            image_features = image_features.permute(0, 2, 1).view(-1, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()  
            image_features = image_features.view(-1, self.feat_dim, image_w * image_h).permute(0, 2, 1)  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if args.text_adjust == 'True' and (args.module_rsap_v1 or args.module_visual_guidance):
            logits = image_features.to(device) @ updated_query_features.T.to(device)  
        else:
            logits = image_features.to(device) @ self.text_features.T.to(device)  
        
        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda 

        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)  


        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return (
            logits,
            image_features.permute(0, 2, 1).contiguous().view(-1, self.feat_dim, image_w, image_h),
            reliability_map,
        )
    
    
    def forward_slide(self, img, ori_shape, image_name,args, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        device = next(self.parameters()).device 
        
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape

        out_channels = len(self.query_idx) 
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img)).to(device)  
        count_mat = img.new_zeros((img.shape[0], 1, h_img, w_img))
        reliability_assem = None
        self.last_reliability_map = None
        if args.module_rsap_v1 or args.loss_sdr:
            reliability_assem = torch.zeros(
                (batch_size, 1, h_img, w_img),
                device=device,
                dtype=torch.float32,
            )
        self.i=0
        self.j=0
        for h_idx in range(h_grids):
            self.i+=1
            for w_idx in range(w_grids):
                self.j+=1
                self.h_idx = h_idx
                self.w_idx = w_idx
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit, image_features_featup, crop_reliability = self.forward_feature(
                    crop_img, image_name, args
                )
                logit_patch = crop_seg_logit  
                logit_patch = logit_patch.mean(1)  

                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]  
                    if crop_reliability is not None:
                        crop_reliability = crop_reliability[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[-1] - x2), int(y1),
                                            int(preds.shape[-2] - y2)))

                if reliability_assem is not None:
                    if crop_reliability is None:
                        raise RuntimeError(
                            'Reliability-guided path is enabled but forward_feature returned no reliability map.'
                        )
                    reliability_assem += nn.functional.pad(
                        crop_reliability.float(),
                        (int(x1), int(preds.shape[-1] - x2), int(y1),
                         int(preds.shape[-2] - y2)),
                    )
                
                
                count_mat[ :, :, y1:y2, x1:x2] += 1 
        assert (count_mat == 0).sum() == 0

        preds = preds.to(self.device) / count_mat.to(self.device)
        if reliability_assem is not None:
            reliability_assem = reliability_assem / count_mat.float().clamp_min(1.0)

        W_out, H_out = ori_shape
        logits = nn.functional.interpolate(preds, size=(H_out, W_out), mode='bilinear')
        if reliability_assem is not None:
            full_reliability = F.interpolate(
                reliability_assem,
                size=(H_out, W_out),
                mode='bilinear',
                align_corners=False,
            ).clamp(0.0, 1.0)
            # Strict CTTA uses batch size 1; SDR consumes a detached [H, W] map.
            self.last_reliability_map = full_reliability[0, 0].detach()

        if args.loss_prompt == 'True':
            pred_mask, pred_logit, all_seg_logits = self.postprocess_result(logits.to(logit_patch.device), image_name,args)
            return pred_mask, pred_logit, all_seg_logits  
        else:
            pred_mask, pred_logit = self.postprocess_result(logits.to(logit_patch.device), image_name, args)
            return pred_mask, pred_logit  
            
        
    def postprocess_result(self, logits, filename, args):   

        batch_size = logits.shape[0]   
        num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)

        cls_to_desc = {int(cls_id): [] for cls_id in set(self.query_idx.tolist())}
        for idx, cls_id in enumerate(self.query_idx):
            cls_to_desc[int(cls_id)].append(idx)

        desc_counts = [len(v) for v in cls_to_desc.values()]
        num_desc = min(desc_counts)

        for i in range(batch_size):
            seg_logits_i = logits[i] * self.logit_scale  
            seg_logits = seg_logits_i.softmax(0) 

            if args.loss_prompt == 'True':
                group_seg_logits = []
                for d in range(num_desc):
                    selected_indices = []
                    for cls_id, desc_list in cls_to_desc.items():
                        if d < len(desc_list):
                            selected_indices.append(desc_list[d])
                    if not selected_indices:
                        continue 
                    seg_logits_group = logits[i][selected_indices, :, :]  
                    group_seg_logits.append(seg_logits_group)

            
            if num_cls != num_queries:
                # Aggregate only the prompts belonging to each class instead of
                # materializing a [num_cls, num_queries, H, W] masked tensor.
                # This preserves the original max/mean reduction exactly while
                # avoiding a large full-resolution temporary allocation.
                class_prob_maps = []
                for cls_id in range(num_cls):
                    cls_mask = self.query_idx == cls_id
                    cls_probs = seg_logits[cls_mask]
                    cls_prob = (
                        self.logit_weight * cls_probs.max(dim=0).values
                        + (1 - self.logit_weight) * cls_probs.mean(dim=0)
                    )
                    class_prob_maps.append(cls_prob)
                seg_logits = torch.stack(class_prob_maps, dim=0)

            seg_pred = seg_logits.argmax(0, keepdim=True)  

            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
    
        if args.loss_prompt == 'True':
            return seg_pred, seg_logits, group_seg_logits
        else:
            return seg_pred, seg_logits
 
    
    def process_logit_pred(self, logits, logit_scale):
        batch_size = logits.shape[0]  
        for i in range(batch_size):
            seg_logits_i = logits[i] * logit_scale   #[cls_num*descrip, w_ori, h_ori]
            seg_logits = seg_logits_i.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                class_prob_maps = []
                for cls_id in range(num_cls):
                    cls_mask = self.query_idx == cls_id
                    cls_probs = seg_logits[cls_mask]
                    cls_prob = (
                        self.logit_weight * cls_probs.max(dim=0).values
                        + (1 - self.logit_weight) * cls_probs.mean(dim=0)
                    )
                    class_prob_maps.append(cls_prob)
                seg_logits = torch.stack(class_prob_maps, dim=0)

    
        return seg_logits
    
    def postprocess_entropy(self, topk, seg_logits, image_path, img):
        batch_size = seg_logits.shape[0]
        
        for i in range(batch_size):
            seg_logits_i = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits_i.softmax(0)

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx).to(seg_logits.device)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            probs = F.softmax(seg_logits_i, dim=0)
            entropies = -(probs * torch.log(probs.clamp(min=1e-6)))
            entropy_map = entropies.mean(dim=0)  # [H,W]

            C, H, W = seg_logits.shape
            topk_indices_list = [] 
            for c in range(C):
                mask = seg_pred[0] == c
                if mask.any():
                    entropy_masked = entropy_map.clone()
                    entropy_masked[~mask] = float('inf')
                    flat_entropy = entropy_masked.flatten() 
                    valid_mask = flat_entropy < float('inf')
                    valid_entropies = flat_entropy[valid_mask]
                    valid_indices_flat = torch.where(valid_mask)[0]
                    if len(valid_entropies) > 0:
                        k = min(topk, len(valid_entropies))
                        topk_values, topk_rel_indices = valid_entropies.topk(k, largest=False)
                        topk_indices = valid_indices_flat[topk_rel_indices]
                        avg_entropy = topk_values.mean()
                        if avg_entropy < entropy_map.mean(): 
                            topk_indices_list.append(topk_indices.tolist())
                        else:
                            topk_indices_list.append(None)
                    else:
                        topk_indices_list.append(None)
                else:
                    topk_indices_list.append(None)

        return topk_indices_list

    @torch.enable_grad()
    def forward(self, image, ori_shape, image_name, args, test=False):  
        text_features = self.get_text_features(test=test) 

        if self.text_shift:
            if args.with_templates:
                text_features = self.text_shifter(text_features, test)
            else:
                text_features = self.text_shifter(text_features.squeeze(1), test)

        self.test = test
        self.text_features = F.normalize(text_features, dim=-1) 

        image = image.type(self.dtype).to(self.device)

        if args.loss_prompt == 'True':
            mask_pred, seg_logits_post,all_seg_logits = self.forward_slide(image, ori_shape, image_name,args, stride=112, crop_size=224) 
            return mask_pred, seg_logits_post, all_seg_logits
        else:
            mask_pred, seg_logits_post = self.forward_slide(image, ori_shape, image_name,args, stride=112, crop_size=224) 
            return mask_pred, seg_logits_post 
    
    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b


def get_shift_model(args, classnames):
    if args.test_sets.replace('_sub', '') in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.replace('_sub', '').lower()))
    
    classnames = [name.replace("_", " ") for name in classnames]

    print('args.text_shift', args.text_shift)
    print('args.do_shift', args.do_shift)

    model = TestTimeShiftTuning(args.gpu, classnames, batch_size=None, arch=args.arch,
                            test_set=args.test_sets,
                            init_concepts=args.init_concepts,   
                            per_label=args.per_label,            
                            with_concepts=args.with_concepts,   
                            text_shift=args.text_shift, 
                            do_shift=args.do_shift,    
                            do_scale=args.do_scale, 
                            do_film=args.do_film, 
                            with_templates=args.with_templates, 
                            macro_pooling=args.macro_pooling,
                            with_coop=args.with_coop, 
                            use_susx_feats=args.use_susx_feats,
                            args = args
                        )

    return model

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

def get_cls_idx_multi_prom(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('|')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [clean_string(item.replace('\n', '')) for item in class_names]
    return class_names, class_indices
