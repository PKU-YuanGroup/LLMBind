



### 2. Training

#### 2.1 SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

#### 2.2 
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='./llmbind_dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b" \
  --steps_per_epoch 500 \
  --epochs 10 \
  --batch_size   16 \
  --model_max_length  768 \
  --add_generation_token \
  --add_edit_token \
  --add_video_generation_token \
  --add_audio_generation_token \
  --vqa_sample_rates='2,70,70,70' \
  --vqa_data "gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
 
```
For example:
```
cd  /remote-home/zhubin/LLMBind 
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```
#### LLaVA1-1
```
deepspeed --master_port=24999 --include=localhost:7 train_ds.py \
  --version="liuhaotian/LLaVA-Lightning-7B-v1-1-zb" \
  --dataset_dir='./llmbind_dataset' \
  --vision_pretrained="cache/sam_vit_h_4b8939.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b" \
  --steps_per_epoch 500 \
  --epochs 2 \
  --batch_size   12  \
  --model_max_length  768 \
  --vqa_data="gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
  --vqa_sample_rates='2,70,70,70' \
  --add_generation_token \
  --add_edit_token \
  --add_video_generation_token \
  --add_audio_generation_token 
```
#### LLaVA1-5

`branch no_seg_split`
```
deepspeed --master_port=24999 --include=localhost:7 train_ds.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --dataset_dir='./llmbind_dataset' \
  --vision_pretrained="cache/sam_vit_h_4b8939.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b" \
  --steps_per_epoch 3 \
  --epochs 2 \
  --batch_size   12  \
  --model_max_length  768 \
  --vqa_data="gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
  --vqa_sample_rates='2,70,70,70'  \
  --add_generation_token \
  --add_edit_token \
  --add_video_generation_token \
  --add_audio_generation_token 
```
`branch main(splitseg) `
```
HF_DATASETS_OFFLINES=1 deepspeed --master_port=24990 --include=localhost:4  train_ds_splitseg.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --dataset_dir='./llmbind_dataset' \
  --vision_pretrained="cache/sam_vit_h_4b8939.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b-splitseg" \
  --steps_per_epoch 3 \
  --epochs 2 \
  --batch_size   16  \
  --model_max_length  768 \
  --vqa_data="gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
  --vqa_sample_rates='2,70,70,70'  \
  --add_generation_token \
  --add_edit_token \
  --add_video_generation_token \
  --add_audio_generation_token \
  --lora_r 8 
```

When training is finished, to get the full model weight:
```
cd ./runs/lllmbind-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```
or
```
cd runs/llmbind-7b-splitseg/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```
### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_HF_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --weight="runs/llmbind-7b-splitseg/pytorch_model.bin" \
  --save_path="runs/llmbind-7b-splitseg/hf_weights" \
  --vision_pretrained cache/sam_vit_h_4b8939.pth \
  --add_generation_token \
  --add_edit_token \
  --add_video_generation_token \
  --add_audio_generation_token 
```

### 3. Inference

- To chat with [LLMBind-7B](xxxx):
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version="PATH_TO_SAVED_HF_MODEL"
```

For example:
```
HF_DATASETS_OFFLINES=1 CUDA_VISIBLE_DEVICES=7 python chat.py --version="runs/llmbind-7b-splitseg/hf_weights"
```
or
```
HF_DATASETS_OFFLINES=1 CUDA_VISIBLE_DEVICES=6 python chat_splitseg.py --version="runs/llmbind-7b-splitseg/hf_weights"
```


- To use `bf16` or `fp16` data type for inference:
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version="PATH_TO_SAVED_HF_MODEL" --precision='bf16'
```
- To use `8bit` or `4bit` data type for inference:
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version="PATH_TO_SAVED_HF_MODEL" --precision='fp16' --load_in_8bit
CUDA_VISIBLE_DEVICES=0 python chat.py --version="PATH_TO_SAVED_HF_MODEL" --precision='fp16' --load_in_4bit
```
`Hint: for 13B model, 16-bit inference consumes 30G VRAM with a single GPU, 8-bit inference consumes 16G, and 4-bit inference consumes 9G.`








### Tips  
- set `legacy=True` for tokenizer,
```
tokenizer = transformers.AutoTokenizer.from_pretrained(
        ......
        legacy=True # important
    )
```
- add `--vqa_sample_rates` for different VQA-Tasks.

- set  `special tokens` for different modality-tasks
```
DEFAULT_GEN_START_TOKEN = "<gen>"
DEFAULT_GEN_END_TOKEN = "</gen>"
DEFAULT_VID_GEN_START_TOKEN = "<vid_cap>"
DEFAULT_VID_GEN_END_TOKEN = "</vid_cap>"
DEFAULT_AUD_GEN_START_TOKEN = "<aud_cap>"
DEFAULT_AUD_GEN_END_TOKEN = "</audio_cap>"
DEFAULT_EDIT_START_TOKEN = "<edit>"
DEFAULT_EDIT_END_TOKEN = "</edit>"
```


- `LLMBind_splitseg.py`   **LISAForCausalLM.evaluate**

如果默认输入必须有一个图片的话（可以是torch.zeros(1,3,224,224)），LLMBind.py
如果在训练的时候将纯文本和图片分开的话，需要在LLMbind evaluate的时候修改一下下面代码。
```python
#=================================
'''
condition1: 
tuple(
        [1,301,4096],
        [1,1,4096],
        [1,1,4096],
        .......
        [1,1,4096],
    )
'''
if len(outputs.hidden_states)>0: 
    if outputs.hidden_states[0].shape[1]>1 and outputs.hidden_states[-1].shape[1]==1:
        output_hidden_states = torch.cat(outputs.hidden_states, dim=1)
else:
    output_hidden_states = outputs.hidden_states[-1] 

#=================================
'''
condition2:
tuple:  num_answer*(torch.Size([1, 313--313+num_, 4096]), ) -> 1*(torch.Size([1, 313+num_, 4096]), ) 
'''
output_hidden_states = outputs.hidden_states[-1] 


# hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
if seg_token_idx in output_ids2 and images is not None:
  seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
  seg_token_mask = torch.cat(
      [
          torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
          seg_token_mask,
      ],
      dim=1,
  )

```