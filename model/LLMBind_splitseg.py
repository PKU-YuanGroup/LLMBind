from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def split_kwargs_by_seg_flag_list(
            self,
            images, # (bs, 3, 1024, 1024)
            images_clip, # (bs,3,224, 224  )
            input_ids, # (bs~bs*3, length) 
            labels, # (bs~bs*3, length)
            attention_masks, # (bs~bs*3, length)
            offset, # tensor: List
            masks_list, # (bs, 3, h ,w)
            label_list, # (bs, h, w)
            resize_list, # (bs, 1) #list[list[]] 
            inference, # bool
            seg_flag_list
        ):
            # import ipdb; ipdb.set_trace()
            # reverse seg_flag_list
            seg_flag_reverse_list = [ not flag for flag in seg_flag_list]
            
            images_seg = images[seg_flag_list]  
            
            images_clip_seg = images_clip[seg_flag_list]
            images_clip_notseg = images_clip[seg_flag_reverse_list]
            
            # [0, 1, ,2, 3, 4]
            # 第一个0是不变的，后面其实是conversations的累加
            conversation_nums = []
            for i in range(1, len(offset)):
                conversation_nums.append(offset[i]-offset[i-1])
            conversation_nums = torch.stack(conversation_nums)
            conversation_nums_seg = conversation_nums[seg_flag_list]
            # 对新的seg batch进行conversation累加统计
            conversation_nums_seg = conversation_nums_seg.cumsum(-1)
            # 最一开始的0不能忘记了
            zero_ = torch.tensor(0, dtype = offset.dtype).to(offset.device)
            offset_seg = torch.cat((zero_.unsqueeze(0), conversation_nums_seg), dim=0)
            
            # seg_flag_list是针对sample计数的，从0~bs-1
            # seg_flag_repeat_list是将分割image中有多个分割对象的conversation的数目repeat一下
            # 这样就可以与input_ids, labels, attention_masks对应起来
            seg_flag_repeat_list = []
            for i, seg_flag in enumerate(seg_flag_list):
                if seg_flag == True:
                    seg_conv_nums =  conversation_nums[i]
                    for i in range(seg_conv_nums):
                        seg_flag_repeat_list.append(True)
                else:
                    seg_flag_repeat_list.append(False)
            # 与seg_flag_reverse_list 相比，就是将有多个conversation的分割数据的 seg_flag重复了相应次数。
            seg_flag_repeat_reverse_list = [ not flag for flag in seg_flag_repeat_list]

            input_ids_seg = input_ids[seg_flag_repeat_list]
            input_ids_notseg = input_ids[seg_flag_repeat_reverse_list]

            labels_seg = labels[seg_flag_repeat_list]
            labels_notseg = labels[seg_flag_repeat_reverse_list]

            attention_masks_seg = attention_masks[seg_flag_repeat_list]
            attention_masks_notseg = attention_masks[seg_flag_repeat_reverse_list]
            
            
            
            # offset_seg = offset[0]  offset[seg_flag_list]
            masks_list_seg = [ masks_list[idx]  for idx, seg_flag in enumerate(seg_flag_list) if seg_flag ]
            label_list_seg = [ label_list[idx]  for idx, seg_flag in enumerate(seg_flag_list) if seg_flag ]
            resize_list_seg = [ resize_list[idx]  for idx, seg_flag in enumerate(seg_flag_list) if seg_flag ]

            return (
                ( 
                    images_seg, 
                    images_clip_seg,
                    input_ids_seg,
                    labels_seg,
                    attention_masks_seg,
                    offset_seg,
                    masks_list_seg,
                    label_list_seg,
                    resize_list_seg,
                    inference, 
                ),
                
                (
                    images_clip_notseg,
                    input_ids_notseg,
                    labels_notseg,
                    attention_masks_notseg,
                    inference,     
                )
            )
    

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        seg_flag_list: List[bool] = None,
        **kwargs,
    ):
        # import ipdb; ipdb.set_trace()
        # 将一个batch内的所有kwargs按照是否为分割任务来进行划分。
        seg_kwargs, notseg_kwargs =  self.split_kwargs_by_seg_flag_list(
                                            images,
                                            images_clip,
                                            input_ids,
                                            labels,
                                            attention_masks,
                                            offset,
                                            masks_list,
                                            label_list,
                                            resize_list,
                                            inference,
                                            seg_flag_list
                                        ) 
        ce_loss = torch.tensor(0.).to(images.device)
        mask_bce_loss = torch.tensor(0.).to(images.device)
        mask_dice_loss = torch.tensor(0.).to(images.device)
        mask_loss = torch.tensor(0.).to(images.device)
        loss = torch.tensor(0.).to(images.device)
        #=================================================================
        #=============== conversation or generation or edition data  =====
        # print(f'seg_data_nums:{seg_flag_list.count(True)}, not_seg_data_nums:{seg_flag_list.count(False)}!!!')
        
        (
            images_clip_notseg,
            input_ids_notseg,
            labels_notseg,
            attention_masks_notseg,
            inference,     
        ) = notseg_kwargs
        
        ( 
            images,  # torch.Size([4, 3, 1024, 1024])
            images_clip, # torch.Size([4, 3, 224, 224])
            input_ids, # torch.Size([12, 169])
            labels, # torch.Size([12, 169])
            attention_masks, # torch.Size([12, 169])
            offset, # tensor([ 0,  3,  6,  9, 12], device='cuda:0')
            masks_list, # List: [(3,1068,1600)]*bs
            label_list, # List: [torch.Size([1068, 1600])]*bs
            resize_list, # [[684, 1024], [683, 1024], [768, 1024], [1024, 685]]
            inference,  # False
        ) = seg_kwargs
        #=================================================================
        #======================= no segmentation_data ================== 
        if seg_flag_list.count(True) == 0: 
            # 输入llava里面的直接就是processor处理之后的image就行，不需要encode.
            # image_embeddings = self.get_visual_embs(images) sam encoder!
            #==========================================
            output = super().forward(
                    images=images_clip_notseg,
                    attention_mask=attention_masks_notseg,
                    input_ids=input_ids_notseg,
                    labels=labels_notseg,
                    output_hidden_states=True,
                )
            ce_loss += output.loss
            loss += ce_loss
        #=================================================================
        #======================= segmentation data  ======================
        # 有分割数据，2种情况，情况1：有vqa， 情况2：无vqa
        # import ipdb; ipdb.set_trace()
        if  seg_flag_list.count(True) > 0:  
            image_embeddings = self.get_visual_embs(images) # visual_model is ViT 256x256
            batch_size = image_embeddings.shape[0]
            assert batch_size == len(offset) - 1
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
                dim=1,
            )
            #========================  FIXME =======================
            if inference:
                n_batch = 1
                length = input_ids.shape[0]
                assert images_clip.shape[0] == 1
                images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

                output_hidden_states = []
                for i in range(n_batch):
                    start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                    output_i = super().forward(
                        images=images_clip_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True,
                    )
                    output_hidden_states.append(output_i.hidden_states)
                    torch.cuda.empty_cache()

                output_hidden_states_list = []
                output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
                output_hidden_states_list.append(output_hidden_states_level)
                output_hidden_states = output_hidden_states_list
                output = None

            else:
                images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    images_clip_i = (
                        images_clip[i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    images_clip_list.append(images_clip_i)
                images_clip = torch.cat(images_clip_list, dim=0)
                
                # ============ condition1   has vqa_data ============ 
                if seg_flag_list.count(False)>0:
                    images_clip_mix = torch.cat((images_clip_notseg, images_clip), dim=0)
                    attention_masks_mix = torch.cat((attention_masks_notseg, attention_masks), dim=0)
                    input_ids_mix = torch.cat((input_ids_notseg, input_ids), dim=0)
                    labels_mix = torch.cat((labels_notseg, labels), dim=0)
                    sep_size = images_clip_notseg.shape[0] # jump
                # ============ condition2   no vqa_data ============ 
                else:
                    images_clip_mix =   images_clip
                    attention_masks_mix =   attention_masks
                    input_ids_mix =   input_ids
                    labels_mix =   labels
                    sep_size = 0 

                output_mix = super().forward(
                    images=images_clip_mix,
                    attention_mask=attention_masks_mix,
                    input_ids=input_ids_mix,
                    labels=labels_mix,
                    output_hidden_states=True,
                )
                #============ ce loss  ==================
                ce_loss = output_mix.loss
                output_hidden_states = output_mix.hidden_states # tuple: len=33, 33*17*(324,4096) 
                # =======  extract seg_data hidden_states from mix_data ======
                # sep_size is crutial
                output_hidden_states = tuple([output_hidden_state[sep_size:] for output_hidden_state in output_hidden_states])

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1])) # output_hidden_states[-1].shape torch.Size([bs_notseg+bsseg, 825, 4096]) 

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) # (3, 431, 256)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            seg_token_offset = seg_token_offset[offset]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

            # model_output = output
            gt_masks = masks_list

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }

            # output = output_mix.logits[sep_size:]
            
            # 因为ce_loss_weight=1，所以正好
            # ce_loss = ce_loss * self.ce_loss_weight  
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]

                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss
            # (  (ce_loss), (w1* ce_loss_seg, w2*mask_loss))
            loss += ce_loss 
            loss += mask_loss

        # 返回的是 conversation, generation, edition, segmentation部分的loss之和
        # print(f'=========seg_data nums:{seg_flag_list.count(True)},vqa_data nums:{seg_flag_list.count(False)}==========', {
        #     "loss": f"{loss:.4f}",
        #     "ce_loss": f"{ce_loss:.4f}",
        #     "mask_bce_loss": f"{mask_bce_loss:.4f}",
        #     "mask_dice_loss": f"{mask_dice_loss:.4f}",
        #     "mask_loss": f"{mask_loss:.4f}",
        # })
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
