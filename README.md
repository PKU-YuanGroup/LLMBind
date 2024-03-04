
<p align="center">
    <img src="https://upload.cc/i1/2024/02/26/kI0bhL.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2401.15947">LLMBind: A Unified Modality-Task Integration Framework</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">
    



[![arXiv](https://img.shields.io/badge/Arxiv-2401.15947-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2401.15947) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/LLMBind/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPKU-YuanGroup%2FMoE-LLaVA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/PKU-YuanGroup/MoE-LLaVA?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/LLMBind/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/PKU-YuanGroup/MoE-LLaVA?color=success&label=Issues)](https://github.com/PKU-YuanGroup/LLMBind/issues?q=is%3Aissue+is%3Aclosed)  <br>
</h5>

<details open><summary>💡 I also have other multi-modal projects that may interest you ✨. </summary><p>
<!--  may -->

> [**LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment**](https://arxiv.org/abs/2310.01852) <br>
> Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, Li Yuan <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/LanguageBind)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/LanguageBind.svg?style=social)](https://github.com/PKU-YuanGroup/LanguageBind)  [![arXiv](https://img.shields.io/badge/Arxiv-2310.01852-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2310.01852) <br>


> [**Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**](https://arxiv.org/abs/2311.10122) <br>
> Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, Li Yuan <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Video-LLaVA)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Video-LLaVA.svg?style=social)](https://github.com/PKU-YuanGroup/Video-LLaVA) [![arXiv](https://img.shields.io/badge/Arxiv-2311.10122-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.10122) <br>

</p></details>


## 📣 News
* **[2024.01.27]**  🤗 Huggingface demo will be available soon! Welcome to **watch** 👀 this repository for the latest updates.

## 😮 Highlights

LLMBind demonstrates promising results in advancing the development of human-like MLLM and AI agents.
### 🔥 A unified model integration framework
- We design a **unified model integration framework** that expands task-specific tokens for diverse modality tasks, thus easily integrating different tasks into a unified LLM, where we introduce the MoE technique in our framework to better handle diverse modality tasks.
<p align="center">
<img src="assets/intro0.jpg" width=55%>
</p>

### 🔥 A unified MLLM with various modality tasks
- We propose **a unified MLLM** that is compatible with **various modality tasks**, including image segmentation, image generation, image editing, video generation, and audio generation.

<p align="center">
<img src="assets/intro.jpg" width=65%>
</p>


### 🔥 Interactive generation and editing datasets
- To facilitate the development of user-friendly interactive tasks, we construct a dataset of 400k interactive generation and editing multi-turn dialogues using ChatGPT. We plan to release this dataset as an open resource to foster collaborative advancements in this field.

<p align="center">
<img src="assets/intro.jpg" width=65%>
</p>



## ⚙️ Requirements and Installation
We recommend the requirements as follows.
* Python == 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* **Transformers == 4.37.0**
* **Tokenizers==0.15.1**
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/LLMBind
cd MoE-LLaVA
conda create -n moellava python=3.10 -y
conda activate moellava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Below are optional. For Qwen model.
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# If the version of flash-attn is higher than 2.1.1, the following is not needed.
# pip install csrc/rotary
```

## Dataset preparation
### 1. Reasoning segmentation dataset: 
[ReasonSeg](https://github.com/dvlab-research/LISA#dataset)
Download them from the above links, and organize them as follows.

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
│   │   └── llava_instruct_150k.json
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

### 2. Interactive generation and editing dataset: 
[Download_error](xxxxxx)

Download them from the above links, and organize them as follows.
```
├── llmbind_dataset
│   ├── interactive_dataset
│   │   ├── audio_t2x_format.json
│   │   ├── image_t2x_format.json
│   │   ├── video_t2x_format.json
│   │   └── gpt_interactive_generation_and_editing_format.json
```


## 🗝️ Training & Validating
The training & validating instruction is in [TRAIN.md](docs/TRAIN.md) & [EVAL.md](docs/EVAL.md).

## 💡 Customizing your LLMBind
The instruction is in [CUSTOM.md](docs/CUSTOM.md).

## 😍 Visualization
The instruction is in [VISUALIZATION.md](docs/VISUALIZATION.md).

## 🙌 Related Projects
* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) This framework empowers the model to efficiently utilize the united visual tokens.
* [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) An open source five modalities language-based retrieval framework.

## 👍 Acknowledgement
* [LLaVA](https://github.com/haotian-liu/LLaVA) The codebase we built upon and it is an efficient large language and vision assistant.

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/LLMBind/blob/main/LICENSE) file.
* The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.



## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.



```BibTeX
@article{zhu2023languagebind,
  title={Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment},
  author={Zhu, Bin and Lin, Bin and Ning, Munan and Yan, Yang and Cui, Jiaxi and Wang, HongFa and Pang, Yatian and Jiang, Wenhao and Zhang, Junwu and Li, Zongwei and others},
  journal={arXiv preprint arXiv:2310.01852},
  year={2023}
}
```



## ✨ Star History
[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/MoE-LLaVA&type=Date)](https://star-history.com/#PKU-YuanGroup/MoE-LLaVA&Date)


## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/LLMBind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/MoE-LLaVA" />
</a>
