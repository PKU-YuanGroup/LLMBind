cd /remote-home/zhubin/LLMBind 
cd runs/llmbind-7b-splitseg_bs12_e40/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin

CUDA_VISIBLE_DEVICES=7 python merge_lora_weights_and_save_hf_model.py \
   --version="liuhaotian/llava-v1.5-7b" \
  --weight="runs/llmbind-7b-splitseg_bs12_e40/pytorch_model.bin" \
  --save_path="runs/llmbind-7b-splitseg_bs12_e40/hf_weights" \
  --vision_pretrained cache/sam_vit_h_4b8939.pth \
  --add_generation_token \
  --add_edit_token \
  --add_video_generation_token \
  --add_audio_generation_token 


HF_DATASETS_OFFLINES=1 CUDA_VISIBLE_DEVICES=7 python chat_splitseg.py --version="runs/llmbind-7b-splitseg_bs12_e40/hf_weights"