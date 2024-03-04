### Training
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="llmbind-7b"
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