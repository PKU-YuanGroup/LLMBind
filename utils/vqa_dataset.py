import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        # vqa_data="llava_instruct_150k",
        # 650k, 400k, 9k, 9k, 9k
        vqa_data="llava_v1_5_mix665k||minidalle3_v3_392272_filter||audio_t2x_format||image_t2x_format||video_t2x_format",
        vqa_sample_rates='1,2,70,70,70'
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        DATA_DIR = os.path.join(base_image_dir, "vqa_dataset")

        # import ipdb; ipdb.set_trace()
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")

        self.vqa_sample_rates = [int(x) for x in vqa_sample_rates.split(',')]
        self.vqa_data = []
        vqa_datalist = vqa_data.split('||')
        for idx, vqa_data_name in enumerate(vqa_datalist):
            with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data_name))) as f:
                vqa_data = json.load(f)
                self.vqa_data.extend(vqa_data*self.vqa_sample_rates[idx])

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.vqa_data) - 1)
            item = self.vqa_data[idx]            
            if "image" in item:
                image_path = os.path.join(self.vqa_image_root, item["image"])
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ori_size = image.shape[:2]
                image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][
                    0
                ]  # preprocess image for clip
                image = self.transform.apply_image(image)  # preprocess image for sam
                resize = image.shape[:2]
                image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()) # ->(3,1024,1024)
            else:
                # =====================  FIXME =====================
                image_path = None
                image = torch.zeros(3, 1024, 1024)
                crop_size = self.clip_image_processor.crop_size
                image_clip = torch.zeros(3, crop_size['height'], crop_size['width'])
                resize =  (878, 1024)
                ori_size =   (549, 640)
                
            conv = conversation_lib.default_conversation.copy()
            source = item["conversations"]
            source = preprocess_multimodal(
                source,
                mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
            )
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conversations = []
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

            questions = conversations
            sampled_classes = conversations
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
            return (
                image_path,
                image,
                image_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
            )
        except Exception as e:
            print(f'=========vqa_dataset error:{e}=======')
            return self.__getitem__(0)
