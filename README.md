# Multimodal-Table-Understanding

[![paper](https://img.shields.io/badge/Paper-ACL_2024-red)](https://arxiv.org/abs/2406.08100) [![dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/SpursgoZmy/MMTab) [![table_llava_7b](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow)](https://huggingface.co/SpursgoZmy/table-llava-v1.5-7b) [![llava_version](https://img.shields.io/badge/Code_Base-🌋_LLaVA_v1.5-yellow)](https://github.com/haotian-liu/LLaVA)

## 1.Introduction

<img src="./readme_images/radar_graph_v4.jpg" width = "280" height = "320" align=right />

Although great progress has been made by recent LLM-based table understanding methods, they rely heavily on the premise that given tables must be converted into a certain text sequence (such as Markdown or HTML) to serve as model input. However, it is difficult to access high-quality textual table representations in some real-world scenarios like scanned documents and webpage screentshots, and table images are much more accessible. Therefore, how to directly understand tables using intuitive visual information
is a crucial and urgent challenge for developing more practical applications.

Facing the above challenge, we propose the multimodal table understanding problem, where the model is required to generate correct responses to different table-related requests (e.g., questions) in an end-to-end fashion based on the table image. Correspondingly, we construct **MMTab**, the first open-source large-scale dataset for multimodal table understanding problem, which can support both the training and evaluation of generalist MLLMs towards multimodal table understanding. Based on the curated MMTab dataset, we develop a versatile tabular MLLM named **Table-LLaVA** with an enhanced two-stage training paradigm of LLaVA v1.5. Table-LLaVA beats
strong MLLM baselines on 17 held-in and 6 held-out benchmarks, and is even competitive with the powerful GPT-4V on 14 benchmarks under a subset of test samples. The right figure shows an intuitive comparison of Table LLaVA 7B and existing MLLMs on various multimodal table understanding benchmarks.

## 2. Dataset Description
We constructed MMTab based on 14 publicly available table datasets of 8 domains. We carefully design scripts to convert original textual tables in these datasets into table images highlighting a broad coverage of table structures and styles, and transform all task-specific samples into multimodal instruction-tuning samples with a unified format of ```<table image, input request,
output response>```. The resulting dataset contains three parts and can be downloaded from the [Hugging Face Dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab). During the dataset
construction, data augmentations at multiple levels (e.g., table-level, task-level) were adopted to further improve the data diversity.

| Dataset Split | #Table Images | #Samples |
| :---: | :---: | :---: |
| **MMTab-pre** | 97K | 150K table recognition samples for pre-training |
| **MMTab-instruct** | 82K | 232K samples of 14 table-based tasks for instruction-tuning |
| **MMTab-eval** | 23K | 45K samples of 17 held-in benchmarks and 4K samples of 7 held-out benchmarks for evaluation |

Dataset examples are shown in the following figure and more examples are shown in the Appendix A in the original paper.

<div align=center>
<img src="./readme_images/dataset_example_4.jpg" width = "800" height = "360" align=center />
</div>

## 3. Model Weights
Table LLaVA follows the LLaVA v1.5 architecture, with [CLIP-ViT-L-336px](https://huggingface.co/openai/clip-vit-large-patch14-336) as the visual encoder (336*336 image resolution), [Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) or [Vicuna-v1.5-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5) as the base LLM and a two-layer MLP as the vision-language connector. We use the code base of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main) for model training and inference. Thus, Table LLaVA can be used in the same way as the normal LLaVA v1.5 model. The saved model checkpoints can be downloaded from the following Hugging Face Repository:

| Version | Size | Schedule | Base LLM | Vision Encoder | Projection layer | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Table LLaVA | 7B | full_finetune-1_epoch | Vicuna-v1.5-7B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-7b](https://huggingface.co/SpursgoZmy/table-llava-v1.5-7b) |  
| Table LLaVA | 13B | full_finetune-1_epoch | Vicuna-v1.5-13B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-13b](https://huggingface.co/SpursgoZmy/table-llava-v1.5-13b) |  
| pretrained_mm_projector of Table LLaVA 7B | 5M | full_finetune-1_epoch | Vicuna-v1.5-7B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-pretrained_mm_projector](https://huggingface.co/SpursgoZmy/table-llava-v1.5-pretrained_mm_projector/tree/main/llava-v1.5-7b-with-table-pretrain) |
| pretrained_mm_projector of Table LLaVA 13B | 5M | full_finetune-1_epoch | Vicuna-v1.5-13B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-pretrained_mm_projector](https://huggingface.co/SpursgoZmy/table-llava-v1.5-pretrained_mm_projector/tree/main/llava-v1.5-13b-with-table-pretrain) |

## 4. Training
### 4.1 Training Data and Hyperparameters
Table LLaVA training consists of two stages: (1) Pre-training stage: the vision-language connector (a two-layer MLP) is trained to connect the frozen pretrained vision encoder (ViT) to the frozen LLM (Vicuna v1.5); (2) Instruction-tuning stage: the vision-language connector and the base LLM are trained to follow multimodal instructions.

The training data of each stage is shown below:

| Training Stage | Data Description | Data Size | Hugging Face Dataset |
| :---: | :---: | :---: | :---: | 
| Pre-training | 558K original LLaVA-1.5 pre-training data | 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
|              | 150K table recognition data (MMTab-pre) | 150K | [MMTab-pre_pretrain_data_llava_format_150K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab) |
| Instruction Fine-tuning | 665K original LLaVA-1.5 fine-tuning data | 665K | [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
|              | 232K multimodal instruction tuning data of 14 tabular tasks (MMTab-instruct) | 232K | [MMTab-instruct_sft_data_llava_format_232K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab) |

The merged pre-training and instruction fine-tuning data in the LLaVA data format can be found in the [MMTab dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab), 
i.e., ```enhanced_llava_pretrain_data_708K.json``` and ```enhanced_llava_sft_data_898K.json```, which can be directly used to train Table LLaVA.

The hyperparameters used in pretraining and finetuning are provided below. We use a similar set of hyperparameters as LLaVA v1.5 except that we increased the
max sequence length from 2048 to 2560 to accommodate longer text sequences.


## 5. Inference

## 6. Evaluation

## 7. Limitations

## TODOs
- [x] Upload the MMTab dataset to Hugging Face.
- [x] Upload the Table LLaVA 7B and 13B model weights to Hugging Face.
- [ ] Upload the code for model training.
- [ ] Upload the code for model inference.
- [ ] Upload the code for evaluation.

## Citation
```bibtex
@misc{zheng2024multimodal,
      title={Multimodal Table Understanding}, 
      author={Mingyu Zheng and Xinwei Feng and Qingyi Si and Qiaoqiao She and Zheng Lin and Wenbin Jiang and Weiping Wang},
      year={2024},
      eprint={2406.08100},
      archivePrefix={arXiv},
      }
}
```
