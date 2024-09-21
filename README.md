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
<img src="./readme_images/dataset_example_4.jpg" width = "800" height = "380" align=center />
</div>

## 3. Model Weights
Table LLaVA follows the LLaVA v1.5 architecture, with [CLIP-ViT-L-336px](https://huggingface.co/openai/clip-vit-large-patch14-336) as the visual encoder (336*336 image resolution), [Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) or [Vicuna-v1.5-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5) as the base LLM and a two-layer MLP as the vision-language connector. The saved model checkpoints can be downloaded from the following Hugging Face Repository:

| Version | Size | Schedule | Base LLM | Vision Encoder | Projection layer | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Table LLaVA | 7B | full_finetune-1_epoch | Vicuna-v1.5-7B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-7b](https://huggingface.co/SpursgoZmy/table-llava-v1.5-7b) |  
| Table LLaVA | 13B | full_finetune-1_epoch | Vicuna-v1.5-13B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-13b](https://huggingface.co/SpursgoZmy/table-llava-v1.5-13b) |  
| pretrained_mm_projector of Table LLaVA 7B | 5M | full_finetune-1_epoch | Vicuna-v1.5-7B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-pretrained_mm_projector](https://huggingface.co/SpursgoZmy/table-llava-v1.5-pretrained_mm_projector/tree/main/llava-v1.5-7b-with-table-pretrain) |
| pretrained_mm_projector of Table LLaVA 13B | 5M | full_finetune-1_epoch | Vicuna-v1.5-13B | CLIP-ViT-L-336px | MLP-2x | [SpursgoZmy/table-llava-v1.5-pretrained_mm_projector](https://huggingface.co/SpursgoZmy/table-llava-v1.5-pretrained_mm_projector/tree/main/llava-v1.5-13b-with-table-pretrain) |

**Note:** The above Table-LLaVA checkpoints are saved from the original LLaVA repository, which is not directly compatible with the Transformers, i.e., it can not be directly loaded in the way like `LlavaForConditionalGeneration.from_pretrained('SpursgoZmy/table-llava-v1.5-7b')`. This problem is mentioned in this [github issue](https://github.com/SpursGoZmy/Table-LLaVA/issues/6). I will try the provided conversion script to make Table-LLaVa checkpoints become compatible with Transformers and upload new checkpoints to a new hub. But for now, maybe the checkpoints can only be loaded with the LLaVA repository like [this](https://github.com/SpursGoZmy/Table-LLaVA/blob/main/llava/eval/model_vqa.py) instead of directly loading from HuggingFace. Sorry for this inconvenience!

## 4. Training
### 4.1 Environment Setup
We use the code base of LLaVA v1.5 for model training and inference. Thus, Table LLaVA can be used as the normal LLaVA v1.5 model and the environment can be installed in a similar way. Note that our code base is downloaded in December 2023 and maybe not the latest. Please refer to the official [LLaVA v1.5 github](https://github.com/haotian-liu/LLaVA/tree/main) for its latest update.

1. Clone this repository and navigate to Table-LLaVA folder
```bash
git clone https://github.com/SpursGoZmy/Table-LLaVA.git
cd Table-LLaVA
```

2. Install Package
```Shell
conda create -n table_llava python=3.10 -y
conda activate table_llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### 4.2 Training Data and Hyperparameters
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

Table LLaVA was trained on 8 A800 GPUs with 80GB memory. We use a similar set of hyperparameters as LLaVA v1.5 except that we increased the
max sequence length from 2048 to 2560 to accommodate longer text sequences. The hyperparameters used in pretraining and finetuning are provided below. 

| Stage | Trained Weights | Global Batch Size | Learning rate | Epochs | Max length | Weight decay | warmup ratio | Deepspeed Stage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pre-training | vision-language connector | 256 | 1e-3 | 1 | 2560 | 0 | 0.03 | ZeRO-2 |
| Instruction Fine-tuning | base LLM and vision-language connector | 128 | 2e-5 | 1 | 2048 | 0 | 0.03 | ZeRO-3 |

### 4.3 Pre-training

1. Download the original images for LLaVA v1.5 pretraining, i.e., ```images.zip``` from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main). Put it under ```./LLaVA-Pretrain/images``` and unzip it.
2. Download ```MMTab-instruct_table_images_82K.zip``` and ```MMTab-pre_table_images_part_2_16K.zip``` from [MMTab dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab/tree/main). Put them under ```./LLaVA-Pretrain/images``` and unzip them. Rename the ```IID_train_image``` dir to ```table_pretrain_part_1```.
3. Download ```enhanced_llava_pretrain_data_708K.json``` from [MMTab dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab/tree/main) to ```./LLaVA-Pretrain```.
4. The resulting data should be organized as follows:

```
LLaVA-Pretrain
├── images
│   ├── table_pretrain_part_1
|   ├── table_pretrain_part_2
|   ├── 00453
|   ├── 00019
|   ├── ...
|   └── 00095
└── enhanced_llava_pretrain_data_708K.json
```

5. Training script with DeepSpeed ZeRO-2: [`pretrain_table_llava.sh`](https://github.com/SpursGoZmy/Table-LLaVA/blob/main/scripts/v1_5/table_llava_scripts/pretrain_table_llava.sh). If you cannot automaticly download the base Vicuna v1.5 and ViT model through HuggingFace, you can download these models manually and set corresponding command-line parameters (```model_name_or_path``` and ```vision_tower```) to the local model paths. Once the pre-training is finished, the trained vision-language projector will be saved at the specified ```output_dir```.

### 4.4 Fine-tuning

1. Create 5 new folders under ```./LLaVA-Finetune/images``` whose names are ```coco```, ```gqa```, ```ocr_vqa```, ```textvqa``` and ```vg```, respectively. Follow instructions from [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) to download images from these 5 datasets for LLaVA v1.5 fine-tuning. Put the zip files in the corresponding folders and unzip them. 
2. Download ```MMTab-instruct_table_images_82K.zip``` from [MMTab dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab/tree/main). Put it under ```./LLaVA-Finetune/images/table_instructV``` and unzip it. Rename the resulting ```IID_train_image``` dir to ```images```.
3. Download ```enhanced_llava_sft_data_898K.json``` from [MMTab dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab/tree/main) to ```./LLaVA-Finetune```.
4. The resulting data should be organized as follows:

```
LLaVA-Finetune
├── images
│   ├── coco
|   |   └── train2017
|   ├── gqa
|   |   └── images
|   ├── ocr_vqa
|   |   └── images
|   ├── textvqa
|   |   └── train_images
|   ├── vg
|   |   ├── VG_100K
|   |   └── VG_100K_2
|   ├── table_instructV
|   |   └── images
└── enhanced_llava_sft_data_898K.json
```

5. Training script with DeepSpeed ZeRO-3: [`continue_sft_table_llava.sh`](https://github.com/SpursGoZmy/Table-LLaVA/blob/main/scripts/v1_5/table_llava_scripts/continue_sft_table_llava.sh). Set the ```pretrain_mm_mlp_adapter``` parameter to the path of your pre-trained vision-language projector, such as ```./pretrained_mm_projector/llava-v1.5-7b-with-table-pretrain/mm_projector.bin```. The trained table llava model will be saved at the specified ```output_dir```.

## 5. Inference
The inference data should be stored in the LLaVA's jsonl format. Each line in the input file corresponds to an input sample, which is a JSON string (generated by `json.dumps()`) of a Python dict. The sample format should look like:

```python
{     "question_id": "TSD_test_item_17", # item_id
      "image": "TABMWP_24663.jpg", # corresponding image file
      "text": "This image displays a table. Could you provide me ...", # input text
      "category": "TABMWP_for_TSD" # {dataset_name}_for_{task_type}, which can be used to separate data of different benchmarks.
}
```

For inference on the MMTab-eval, download the 49K MMTab-eval test samples in the jsonl format ([MMTab-eval_test_data_49K_llava_jsonl_format.jsonl](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/MMTab-eval_test_data_49K_llava_jsonl_format.jsonl)) and its image files ([MMTab-eval_table_images_23K.zip](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/MMTab-eval_table_images_23K.zip)). Then create a folder named 'LLaVA-Inference' and organize the data as follows:

```
LLaVA-Inference
├── MMTab-eval_test_data_49K_llava_jsonl_format.jsonl
└── all_test_image
```

Inference on multi-GPU: [`start_multicard_inference.sh`](https://github.com/SpursGoZmy/Table-LLaVA/blob/main/scripts/v1_5/table_llava_scripts/start_multicard_inference.sh). You can also inference on your own data. Remember adjust parameters like '`question-file`' (input file path), '`image-folder`' (image folder path) in the [`table_llava_inference.sh`](https://github.com/SpursGoZmy/Table-LLaVA/blob/main/scripts/v1_5/table_llava_scripts/table_llava_inference.sh). The inference results (`merge.jsonl`) will be stored in the path of the '`answers-file`' parameter, e.g., `./eval_results/answers/MMTab_eval/table-llava-v1.5-7b/merge.jsonl`.

With the offical inference script, the inference result format in the `merge.jsonl` should look like:

```python
{      'question_id': 'TABMWP_8', # item_id
       'prompt': 'Problem: \nHannah baked cookies each day ...', # input_prompt
       'text': 'Find the numbers in the table.\n\nSaturday: ...', # model_output
       'answer_id': 'jELcxSPcXHBj3xvHfm5r8T', # answer_id
       'model_id': 'table-llava-7b', # model_id
       'category': 'TABMWP_for_TQA'
} # item category
```

## 6. Evaluation
The evaluation scripts are stored in the `MMTab-eval_evaluation` folder. First, `cd MMTab-eval_evaluation` and `pip install -r eval_requirements.txt` to install necessary packages like ['Sacrebleu'](https://github.com/mjpost/sacrebleu) for evaluation. For table recognition task, we use the [PubTabNet's TEDS computation script](https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src) for evaluation. Then, download the MMTab-eval test data ([MMTab-eval_test_data_49K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/MMTab-eval_test_data_49K.json)) and test tables ([MMTab-eval_test_tables_23K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/MMTab-eval_test_tables_23K.json)), and put them into the `MMTab-eval_evaluation` folder together with the LLaVA's inference result (`merge.jsonl`). Use the [MMTab_evaluation.ipynb](https://github.com/SpursGoZmy/Table-LLaVA/blob/main/MMTab-eval_evaluation/MMTab_evaluation.ipynb) notebook for automatic evaluation.

For the evaluation on the ToTTo test set, you need to organize the model output into a txt file and upload it to the offical [ToTTo leaderboard](https://github.com/google-research-datasets/ToTTo#leaderboard-submission). 

## 7. Limitations

## TODOs
- [x] Upload the MMTab dataset to Hugging Face.
- [x] Upload the Table LLaVA 7B and 13B model weights to Hugging Face.
- [x] The code for model training.
- [x] The code for model inference.
- [x] The code for evaluation.

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
