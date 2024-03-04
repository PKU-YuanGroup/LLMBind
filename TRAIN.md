#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Tips  
- set `legacy=True` for tokenizer,
```
tokenizer = transformers.AutoTokenizer.from_pretrained(
        ......
        legacy=True # important
    )
```
- add `--vqa_sample_rates` for different VQA-Tasks.




### Training
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='./llmbind_dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b" \
  --steps_per_epoch 500 \
  --batch_size   16 \
  --model_max_length 1024 \
  --vqa_data="gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
  --vqa_sample_rates='2,70,70,70' 
```
For example:
```
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
  --batch_size   16  \
  --model_max_length 1024 \
  --vqa_data="gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
  --vqa_sample_rates='2,70,70,70' 
```
#### LLaVA1-5
```
deepspeed --master_port=24999 --include=localhost:7 train_ds.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --dataset_dir='./llmbind_dataset' \
  --vision_pretrained="cache/sam_vit_h_4b8939.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b" \
  --steps_per_epoch 500 \
  --batch_size   16  \
  --model_max_length 1024 \
  --vqa_data="gpt_interactive_generation_and_editing_format||audio_t2x_format||image_t2x_format||video_t2x_format" \
  --vqa_sample_rates='2,70,70,70' 
```

When training is finished, to get the full model weight:
```
cd ./runs/lllmbind-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```
### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --weight="llmbind-7b/pytorch_model.bin" \
  --save_path="./LLMBind-7B" \
  --vision_pretrained cache/sam_vit_h_4b8939.pth \
  --add_video_generation_token \
  --add_audio_generation_token
```