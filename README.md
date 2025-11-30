# ProSeg

![Logo](assets/image.png)


> Probabilistic Modeling of Multi-rater Medical Image Segmentation for Diversity and Personalization

_Work in progress_

## Overview
ProSeg models both expert preferences and boundary ambiguity to produce segmentation that is simultaneously diverse and personalized. Two latent variables capture rater style and image uncertainty, with conditional distributions estimated via variational inference. Sampling from these distributions yields expert-specific predictions while preserving diversity across outputs. Experiments on nasopharyngeal carcinoma (NPC) and LIDC-IDRI lung nodule datasets set a new state of the art for multi-rater medical image segmentation.

## Resources
- Project page: https://ai4mol.github.io/projects/ProSeg/
<!-- - Paper: https://doi.org/10.24963/ijcai.2025/1089 -->

## Status & Roadmap
- ✅ Release the main code and dataset
- ⬜ Reconstruct the code structure and release the evaluation pipeline


## At a Glance
- Dual latent variables for rater preference and boundary ambiguity
- Variational inference for conditional distributions over annotations
- Generates expert-personalized yet diverse segmentations
- Validated on NPC and LIDC-IDRI multi-rater datasets

<!--
## Citation
If you find this work useful in your research, please consider citing the paper above.
-->
