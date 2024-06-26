# NLMTrack
This project is the code for "Nature Language Model for Thermal Infrared Tracking."

## Abstract
Thermal infrared tracking is an essential topic in computer vision tasks because of its advantage of all-weather imaging.
However, most conventional methods utilize only hand-crafted features, while deep learning-based correlation filtering methods are limited by simple correlation operations. Transformer-based methods ignore temporal and coordinate information, which is critical for TIR tracking that lacks texture and color information. 
In this paper, we apply natural language modeling to TIR tracking and propose a novel model called NLMTrack, a coordinate sequence generation-based TIR object tracking model. NLMTrack simplifies the TIR tracking pipeline, where the encoder unifies feature extraction and feature fusion, 
and the complex classification and regression heads are discarded. To address the challenge of low detail and low contrast in TIR images, on the one hand, we design a multi-level progressive fusion module that enhances the semantic representation and incorporates multi-scale features. 
On the other hand, the decoder combines the TIR features and the coordinate sequence features using a causal transformer to generate the target sequence step by step. 
Moreover, we explore an adaptive loss aimed at elevaing tracking accuracyand and a simple template update strategy to accommodate the target's appearance variations. Experiments show that NLMTrack achieves state-of-the-art performance on multiple benchmarks.


## Evaluation results on the LSOTB-TIR benchmark
<figure>
  <img src="./tracking/EVALUATION RESULTS.png" alt="table">
  <figcaption style="text-align: center;"></figcaption>
</figure>

## Test
1. Please download the LSOTB-TIR evaluation dataset, PTB-TIR dataset, and VOT-TIR2015 dataset.
2. Configure the path in  `lib/test/evaluation/local.py` and `lib/test/parameter/nlmtrack.py`.
3. Run `tracking/test.py` for testing
4. Evaluation on the [LSOTB-TIR benchmark](https://github.com/QiaoLiuHit/LSOTB-TIR), [PTB-TIR benchmark](https://github.com/QiaoLiuHit/PTB-TIR_Evaluation_toolkit), and [VOT benchmark](https://github.com/votchallenge/toolkit-legacy)

## Train
1. Preparing your training data(GOT-10K dataset, LSOTB-TIR training dataset)
2. Configure the path in `lib/train/admin/local.py`
3. Run `lib/train/run_training.py`
