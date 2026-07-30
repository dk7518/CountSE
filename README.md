# CountSE+: Contrastive Hard-Negative Mining & Privileged Distillation for Open-set Object Counting

> This repository is a fork of [CountSE (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.pdf), extended with an adapter-based contrastive learning module that mines hard negatives automatically and, optionally, distills privileged human-annotated exemplar information at training time. See [What's New](#whats-new-in-this-fork) below for details on our additions.

Original authors: Shuai Liu, Peng Zhang, Shiwei Zhang, Wei Ke\*
Extension: current repo

Official PyTorch implementation for CountSE. Please check the [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.pdf) for details.

## CountSE Architecture

<img src=img/architecture.jpg width="100%"/>

## Contents
* [🆕 What's New in This Fork](#whats-new-in-this-fork)
* [📦 Preparation](#preparation)
* [🎯 Inference](#inference)
* [🚀 Training](#training)
* [🧪 Ablation Options](#ablation-options)
* [📜 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)

## What's New in This Fork

CountSE's exemplar selection (SES + CEF) scores similarity between image patches and text using `ContrastiveEmbed`, a fixed, parameter-free dot product. This means the model has no learned capacity to distinguish the target object from visually similar background or distractor objects — a limitation the original paper identifies as its main failure mode (background confusion).

This fork adds four training-time-only components on top of the original architecture, **with zero change to the inference interface** (no extra exemplar annotation, no extra forward pass):

| Component | What it does | When it runs |
|---|---|---|
| **Exemplar Adapter** | A lightweight residual MLP (zero-init) inserted before similarity scoring, giving SES a learnable projection instead of a fixed dot product. | Training + inference (always) |
| **Hard-Negative Mining** | Reuses CEF's spectral clustering: the discarded (non-largest) clusters — already computed, no extra annotation — become automatic hard-negative candidates. | Training only |
| **GT-Verification Filter** | Before using a discarded cluster as a negative, checks it doesn't spatially overlap a real GT box (to avoid penalizing an accidentally-discarded true positive). | Training only |
| **Contrastive Loss (InfoNCE)** | Pulls the auto-selected exemplar toward its text anchor and pushes it away from verified hard negatives + in-batch easy negatives (other categories' exemplars). | Training only |
| **Privileged-Information Distillation** | Optional. Uses FSC-147's human-annotated exemplar boxes (unused by the original SES/CEF) as a training-only "teacher" signal via RoIAlign, so the auto-selected exemplar is pulled toward what a human would have picked. | Training only, `use_distill=True` |

All of the above are implemented in `models/GroundingDINO/groundingdino.py` (`ExemplarAdapter`, `ExemplarSelector`, `TeacherExemplarEncoder`, `SetCriterion`). A full write-up of the method (with equations) is in [`method_summary.docx`](./method_summary.docx).

Since the negative-mining, GT-filter, and distillation components only affect the training loss, **`models_inference/GroundingDINO/groundingdino.py` only needs the adapter mirrored** (no criterion changes) for evaluating checkpoints trained with this fork.

## Preparation
### 1. Clone Repository

```
git clone git@github.com:<your-github-username>/CountSE.git
cd CountSE
git checkout feature/contrastive-adapter
```

### 2. Dataset Preparation

We use fsc147 to train and test our model. Click the [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) to download it. Then modify the relevant paths of `config/datasets_fsc147_val.json` and `config/datasets_fsc147_test.json` in the root directory.

### 3. Set Up Environment
If you have already run CountGD, you can directly use its environment to run our code.

Install GCC and virtual environment. We used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
sudo apt update
sudo apt install build-essential

conda create -n countse python=3.9.19
conda activate countse
cd CountSE
pip install -r requirements.txt
pip install scikit-learn==1.3.2 seaborn   # not listed in requirements.txt but required by ExemplarSelector
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
```

If you have problems during the installation environment, you can check the [CountGD](https://github.com/niki-amini-naieni/CountGD/issues) or [Open-GrundingDINO](https://github.com/longzw1997/Open-GroundingDino/issues) repository issues for assistance.

### 4. Download Pre-Trained Weights

Download pre-training groudingdino weights

  ```
  wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
  ```

Download Bert

  ```
  python download_bert.py
  ```

## Training
You can use the following command to train the model. The `--output_dir` argument specifies the path where the model will be saved. If you need to resume training, you can use the `--start_epoch` parameter, which defaults to 0.
```
python -u main.py --output_dir ./countse_ckpt -c config/cfg_fsc147_val.py --datasets config/datasets_fsc147_val.json --pretrain_model_path ./pretrained_ckpts/groundingdino_swinb_cogcoor.pth --gpuid 0 --options text_encoder_type=./pretrained_ckpts/bert-base-uncased
```

By default this trains with the adapter + contrastive loss + hard-negative mining + GT-filter enabled (see [Ablation Options](#ablation-options) to turn components on/off).

## Inference
You can download our trained [checkpoint](https://drive.google.com/file/d/1wJaWcrQB_z4LaQwApNChg_xN3DWvqfgP/view?pli=1) to obtain the results in the paper. Update your weight path and you can run inference on a single RTX 3090 using the following command. You can switch test datasets by modifying the `-c` and `--datasets` arguments.

```
python -u main_inference.py --eval --output_dir ./inference_val -c config/cfg_fsc147_test.py --datasets config/datasets_fsc147_test.json --pretrain_model_path ./pretrained_ckpts/checkpoint_best_regular.pth --gpuid 0 --options text_encoder_type=./pretrained_ckpts/bert-base-uncased --crop --remove_bad_exemplar
```

> Note: checkpoints trained with this fork include adapter weights (`exemplar_selector.adapter.*`). Make sure `models_inference/GroundingDINO/groundingdino.py`'s `ExemplarSelector` has the adapter mirrored in, otherwise those weights are silently skipped on load (`strict=False`).

## Ablation Options

Three flags control which added components are active, passed via `--options` to `main.py`:

| Flag | Default | Effect when `False` |
|---|---|---|
| `use_hard_neg` | `True` | Contrastive loss uses only in-batch easy negatives (no CEF hard negatives) |
| `use_gt_filter` | `True` | Hard negatives are used without GT-overlap verification |
| `use_distill` | `False` | Privileged-information distillation is disabled (teacher branch not computed) |
| `contrast_loss_coef` | `0.1` | Set to `0.0` to disable the contrastive loss entirely (adapter-only) |
| `distill_loss_coef` | `0.1` | Weight of the distillation loss when `use_distill=True` |

Suggested ablation ladder:

```
# 1. baseline            -> use the baseline-v0 tag / original code, no options needed
# 2. adapter only
--options contrast_loss_coef=0.0
# 3. + contrastive (easy negatives only)
--options use_hard_neg=False
# 4. + hard negatives (no GT-filter)
--options use_gt_filter=False
# 5. + full (hard negatives + GT-filter)
# (no extra options — this is the default)
# 6. + distillation
--options use_distill=True
```

Add `--debug` to any command to break after 15 iterations for a quick sanity check before committing to a full run.

## Citation

```
@inproceedings{liu2025countse,
  title={CountSE: Soft Exemplar Open-set Object Counting},
  author={Liu, Shuai and Zhang, Peng and Zhang, Shiwei and Ke, Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21536--21546},
  year={2025}
}
```

### Acknowledgements
Our code is based on [CountGD](https://github.com/niki-amini-naieni/CountGD) and [CountSE](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.pdf). If you have any questions about the original model, please contact pppzhang@stu.xjtu.edu.cn. For questions about the extensions in this fork, contact dk7518@g.skku.edu.