# Multimodal-Table-Understanding

[![paper](https://img.shields.io/badge/Paper-ACL_2024-red)](https://arxiv.org/abs/2406.08100) [![dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/SpursgoZmy/MMTab) [![table_llava_7b](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow)](https://huggingface.co/SpursgoZmy/table-llava-v1.5-7b) [![llava_version](https://img.shields.io/badge/🌋_LLaVA-v1.5-yellow)](https://github.com/haotian-liu/LLaVA)

## 1.Introduction

Although great progress has been made by recent LLM-based table understanding methods, they rely heavily on the premise that given tables must be converted into a certain text sequence (such as Markdown or HTML) to serve as model input. However, it is difficult to access high-quality textual table representations in some real-world scenarios like scanned documents and webpage screentshots, and table images are much more accessible. Therefore, how to directly understand tables using intuitive visual information
is a crucial and urgent challenge for developing more practical applications.

Facing the above challenge, we propose the multimodal table understanding problem, where the model is required to generate correct responses to different table-related requests (e.g., questions) in an end-to-end fashion based on the table image. Correspondingly, we construct **MMTab**, the first open-source large-scale dataset for multimodal table understanding problem, based on 14 publicly available table datasets of 8 domains, which can support both the training and evaluation of generalist MLLMs towards multimodal table understanding. Based on the curated MMTab dataset, we develop a versatile tabular MLLM named **Table-LLaVA** with an enhanced two-stage training paradigm of LLaVA v1.5. Table-LLaVA beats
strong MLLM baselines on 17 held-in and 6 held-out benchmarks, and is even competitive with the powerful GPT-4V on 14 benchmarks under a subset of test samples. 

## 2. Dataset Description

## 3. Model Weights

## 4. Training

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
