import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LLMBind_splitseg import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)



import gradio as gr
title_markdown = """
<center>
    <img src="https://camo.githubusercontent.com/e79223ed3f22366c9381673ac49cb288f39639f333cb611552eb90286f538b2f/68747470733a2f2f75706c6f61642e63632f69312f323032342f30322f32362f6b493062684c2e706e67" alt="example" width="200" >
</center>
<div style="text-align:center; font-size:30px;"><b>LLMBind: A Unified Modality-Task Integration Framework</b></div>
<div style="text-align:center;"><a href="https://llava-vl.github.io">Project Page</a> | <a href="https://arxiv.org/abs/2304.08485">Paper</a> | <a href="https://github.com/haotian-liu/LLaVA">GitHub</a> | <a href="https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md">Model</a></div>

# 

The LLMBind is designed to facilitate a user-friendly AI agent for achieving human-like conversation, interactive image, video, and audio generation, as well as interactive image editing and segmentation, among other tasks.
Users can engage with LLMBind through multi-turn language dialogues, during which the model automatically identifies the relevant modality task and selects the appropriate model to accomplish the task.

✨ **Instruction Button**: Direct input of text commands to perform **video, image, audio generation and image editing** tasks.

✨ **Input Image Button (Optional)**:  Used for **image understanding and segmentation** tasks that require an input image.

✨ **Modaility-Task Selection Button (Optional)**: Selecting this button enables our model to perform various modality-tasks **with greater precision**. However, for ease of use, you can simply ignore these buttons.

"""
examples = [
    ["green_horse.jpg", "Change the color of this green horse to white", "Image Editing"],
    ["green_horse.jpg", "What does the man look like in this picture? Please segment the horse if there is one. ", None],
    [None, "I would like to see a video about 'an astronaut riding a horse' now. ", "Video Generation"],
    [None, "I really like dogs. Can you return a segment of their barking sound?", "Audio Generation"],
    [None, "I really like dogs. Can you return a segment of their barking sound?", "Image Generation"],
    [None, "I really like dogs. Can you return a segment of their barking sound?", None]
]

def image_gen(prompt = "a photo of an astronaut riding a horse on mars", save_image_path = './tmp.png'):
    import torch
    from diffusers import StableDiffusionPipeline            
    # model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "cache/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir='./cache')
    # pipe = StableDiffusionPipeline.from_pretrained('cache/stable-diffusion-v1-5', torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]  
    image.save(save_image_path)

def image_edit(prompt="make the mountains snowy", input_image_path='./tmp.png', save_image_path='./tmp_edited.png'):
    import PIL, requests, torch
    from io import BytesIO
    from diffusers import StableDiffusionInstructPix2PixPipeline
    image = PIL.Image.open(input_image_path).convert("RGB").resize((512, 512))
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "cache/timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    image = pipe(prompt=prompt, image=image).images[0]
    image.save(save_image_path)

def audio_gen(prompt = "Techno music with a strong, upbeat tempo and high melodic riffs", save_audio_path = './tmp.wav'):
    from diffusers import AudioLDMPipeline
    import torch; import scipy
    repo_id = "cache/cvssp/audioldm-s-full-v2"
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
    # save the audio sample as a .wav file
    scipy.io.wavfile.write(save_audio_path, rate=16000, data=audio)


def gif_gen_animatediff(prompt="a black dog", save_gif_path='./tmp.gif'):
    import torch
    from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
    from diffusers.utils import export_to_gif
    from IPython.display import Image
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("cache/guoyww/animatediff-motion-adapter-v1-5-2")
    # load SD 1.5 based finetuned model
    model_id = "cache/SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1)
    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    output = pipe(
        prompt=(
            prompt
        ),
        negative_prompt="bad quality, worse quality",
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    frames = output.frames[0]
    export_to_gif(frames, save_gif_path)
    # export_to_video(frames, fps=30,  output_video_path='tmp.mp4')

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=True # wogaide
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()


    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    while True:
        clear_input = input("Do you want to clear the chat history? (yes or no): ")
        while clear_input!= "yes" and clear_input!="no":
            clear_input = input("Do you want to clear the chat history? (yes or no): ")
        if clear_input == 'yes':
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []

        image_input = input("Do you want to input a picture? (yes or no): ")
        while image_input!= "yes" and image_input!="no":
            image_input = input("Do you want to input a picture? (yes or no): ")
        prompt = input("Please input your prompt: ")
        # prompt = 'Where is dog in this image? Please output segmentation mask.'
        
        if image_input == 'yes':
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt # <image>\n+prompt
        elif image_input == "no": 
            prompt =  prompt # <image>\n+prompt
        else:
            raise KeyError
        
        print(f'args.use_mm_start_end={args.use_mm_start_end}!!')
        if args.use_mm_start_end:

            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        
        # image_path = 'imgs/dog_with_horn.jpg'
        if image_input == 'no':
            image_clip = torch.zeros(1,3,224,224).cuda()
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()
            image = None
            resize_list = None
            original_size_list = None
        else:
            image_path = input("Please input the image path: ")
            # image_path = 'imgs/dog_with_horn.jpg'
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))
                continue

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

        # import ipdb; ipdb.set_trace()
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # import ipdb;  ipdb.set_trace()
        # importy
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)


        # =========
        assistant_output = text_output.split(conv.roles[1])[-1]
        print(f"text_output:{text_output}, assistant_output:{assistant_output}!")
        conv.update_last_message(conv.roles[1], assistant_output)
        #======  exec corresponding tasks ====
        import re
        if '<gen>' in assistant_output:
            img_cap =  re.search(r'<gen>(.*?)</gen>', assistant_output).group(1)
            image_gen(prompt=img_cap, save_image_path='./tmp.png')
            print(f'./tmp.png is saved!')
        elif '<edit>' in assistant_output:
            edited_img_cap =  re.search(r'<edit>(.*?)</edit>', assistant_output).group(1)
            image_edit(prompt="make the mountains snowy", input_image_path='./tmp.png', save_image_path='./tmp_edited.png')
            print('./tmp_edited.png is saved!')
        elif '<video_cap>' in assistant_output:
            video_cap =  re.search(r'<video_cap>(.*?)</video_cap>', assistant_output).group(1)
            # animatediff mpdel for video generation
            gif_gen_animatediff(prompt=video_cap, save_gif_path='./tmp.gif')
            # video_gen(prompt=video_cap, save_video_path='./tmp.mp4')
            print(f'./tmp.gif is saved!')
        elif '<audio_cap>' in assistant_output:
            audio_cap =  re.search(r'<audio_cap>(.*?)</audio_cap>', assistant_output).group(1)
            audio_gen(prompt = audio_cap, save_audio_path='./tmp.wav')
            print(f'./tmp.wav is saved!')


        if pred_masks == None:
            print("Answer doesn\'t contain <seg>!")
        else:
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])



'''

HF_DATASETS_OFFLINES=1 CUDA_VISIBLE_DEVICES=7 python chat_splitseg_gradio.py --version="runs/llmbind-7b-splitseg_bs12_e40/hf_weights"

'''
