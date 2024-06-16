
<p align="center">
    <img src="https://upload.cc/i1/2024/02/26/kI0bhL.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2402.14891">LLMBind: A Unified Modality-Task Integration Framework</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">
    



[![arXiv](https://img.shields.io/badge/Arxiv-2401.15947-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.14891) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/LLMBind/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPKU-YuanGroup%2FLLMBind&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/PKU-YuanGroup/LLMBind?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/LLMBind/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/PKU-YuanGroup/LLMBind?color=success&label=Issues)](https://github.com/PKU-YuanGroup/LLMBind/issues?q=is%3Aissue+is%3Aclosed)  <br>
</h5>

<details open><summary>💡 I also have other multi-modal projects that may interest you ✨. </summary><p>
<!--  may -->


> [**Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**](https://arxiv.org/abs/2311.10122) <br>
> Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, Li Yuan <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Video-LLaVA)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Video-LLaVA.svg?style=social)](https://github.com/PKU-YuanGroup/Video-LLaVA) [![arXiv](https://img.shields.io/badge/Arxiv-2311.10122-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.10122) <br>

> [**LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment**](https://arxiv.org/abs/2310.01852) <br>
> Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, Li Yuan <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/LanguageBind)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/LanguageBind.svg?style=social)](https://github.com/PKU-YuanGroup/LanguageBind)  [![arXiv](https://img.shields.io/badge/Arxiv-2310.01852-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2310.01852) <br>



</p></details>


## News
* **[2024.06.15]**  🤗 Huggingface demo will be available soon! Welcome to **watch** 👀 this repository for the latest updates.
* **[2024.06.15]**  🤗 We have release part of our interactive generation and editing dataset in [Huggingface](xxx) and [GitHub](xxxx).


## Highlights

LLMBind demonstrates promising results in advancing the development of human-like MLLM and AI agents.
### 🔥 A unified model integration framework
- We design a **unified model integration framework** that expands task-specific tokens for diverse modality tasks, thus easily integrating different tasks into a unified LLM, where we introduce the MoE technique in our framework to better handle diverse modality tasks.
<p align="center">
<img src="assets/LLMBind_framework_0201.png" width=100%>
</p>

### 🔥 A unified MLLM with various modality tasks
- We propose **a unified MLLM** that is compatible with **various modality tasks**, including image segmentation, image generation, image editing, video generation, and audio generation.

<p align="center">
<img src="assets/conversation-0130.png" width=100%>
</p>


### 🔥 Interactive generation and editing datasets
- To facilitate the development of user-friendly interactive tasks, we construct a dataset of 400k interactive generation and editing multi-turn dialogues using ChatGPT. We plan to release this dataset as an open resource to foster collaborative advancements in this field.

<!-- <div style="display: flex; justify-content: center;">
  <div style="flex: 1; max-width: 40%;">
    <img src="assets/LLMBind_Dataset2.png" width="100%">
  </div>
  <div style="flex: 1; max-width: 40%;">
    <img src="assets/chatgpt-prompt.png" width="100%">
  </div>
</div> -->

## Installation
```bash
git clone https://github.com/PKU-YuanGroup/LLMBind
cd LLMBind
conda create -n llmbind python=3.8 -y
conda activate llmbind
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Dataset preparation

### 1. Interactive generation and editing dataset: 

Download part of them from [Huggingface](xxxx), and organize them as follows.
```
├── llmbind_dataset
│   ├── interactive_dataset
│   │   ├── interactive_audio_t2x_format.json
│   │   ├── interactive_image_t2x_format.json
│   │   ├── interactive_video_t2x_format.json
│   │   └── interactive_generation_and_editing_format.json
```
### 2. Reasoning segmentation & Refering segmentation & VQA dataset: 
[Download](https://github.com/dvlab-research/LISA#dataset) them and organize as follows.
```
├── llmbind_dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_v1_5_mix665k.json
│   ├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   └── vlpart
│       ├── paco
│       │   └── annotations
│       └── pascal_part
│           ├── train.json
│           └── VOCdevkit
```

## Training

## Inference

## Merge LoRA Weight
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
  --version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
  --weight="lisa-7b/pytorch_model.bin" \
  --save_path="./LISA-7B"
```


## Acknowledgement
* [LISA](https://github.com/haotian-liu/LLaVA) The codebase we built upon and it is an efficient large language and vision assistant.
* [LLaVA](https://github.com/haotian-liu/LLaVA) The codebase we built upon and it is an efficient large language and vision assistant.

## License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/LLMBind/blob/main/LICENSE) file.
* The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.


## Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.


```BibTeX
@article{zhu2024llmbind,
  title={LLMBind: A Unified Modality-Task Integration Framework},
  author={Zhu, Bin and Jin, Peng and Ning, Munan and Lin, Bin and Huang, Jinfa and Song, Qi and Pan, Mingjun and Yuan, Li},
  journal={arXiv preprint arXiv:2402.14891},
  year={2024}
}
```



<!-- ## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=PKU-YuanGroup/LLMBind&type=Date)](https://star-history.com/#PKU-YuanGroup/LLMBind&Date) -->

## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/LLMBind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/LLMBind" />
</a> 
