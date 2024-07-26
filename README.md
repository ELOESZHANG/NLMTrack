# NLMTrack
Coordinate-Aware Thermal Infrared Tracking Via Natural Language Modeling."

## Abstract
Thermal infrared (TIR) tracking is pivotal in computer vision tasks due to its all-weather imaging capability. Traditional tracking methods predominantly rely on hand-crafted features, and while deep learning has introduced correlation filtering techniques, these are often constrained by rudimentary correlation operations. Furthermore, transformer-based approaches tend to overlook temporal and coordinate information, which is critical for TIR tracking that lacks texture and color information. In this paper, to address these issues, we apply natural language modeling to TIR tracking and  propose a coordinate-aware thermal infrared tracking model called NLMTrack, which enhances the utilization of coordinate and temporal information. NLMTrack applies an encoder that unifies feature extraction and feature fusion, which simplifies the TIR tracking pipeline. To address the challenge of low detail and low contrast in TIR images, on the one hand, we design a multi-level progressive fusion module that enhances the semantic representation and incorporates multi-scale features. On the other hand, the decoder combines the TIR features and the coordinate sequence features using a causal transformer to generate the target sequence step by step. Moreover, we explore an adaptive loss aimed at elevating tracking accuracy and a simple template update strategy to accommodate the target's appearance variations. Experiments show that NLMTrack achieves state-of-the-art performance on multiple benchmarks. 


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
