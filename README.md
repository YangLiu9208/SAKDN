## Semantics-aware Adaptive Knowledge Distillation Networks for Sensor-to-Vision Action Recognition
Under review by IEEE Transactions on Image Processing    

<a href="https://orcid.org/0000-0002-9423-9252" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">orcid.org/0000-0002-9423-9252</a>

Homepage: [https://yangliu9208.github.io/home/](https://yangliu9208.github.io/home/)

## Abstract
Existing vision-based action recognition is susceptible to occlusion and appearance variations, while wearable sensors can alleviate these challenges by capturing human motion with one-dimensional time-series signals (e.g. acceleration, gyroscope and orientation). For the same action, the knowledge learned from vision sensors (videos or images) and wearable sensors, may be related and complementary. However, there exists significantly large modality difference between action data captured by wearable-sensor and vision-sensor in data dimension, data distribution and inherent information content. In this paper, we propose a novel framework, named Semantics-aware Adaptive Knowledge Distillation Networks (SAKDN), to enhance action recognition in vision-sensor modality (videos) by adaptively transferring and distilling the knowledge from multiple wearable sensors. The SAKDN uses multiple wearable-sensors as teacher modalities and uses RGB videos as student modality. Specifically, we transform one-dimensional time-series signals of wearable sensors to two-dimensional images by designing a gramian angular field based virtual image generation model. Then, we build a novel Similarity-Preserving Adaptive Multimodal Fusion Module (SPAMFM) to adaptively fuse intermediate representation knowledge from different teacher networks. To fully exploit and transfer the knowledge of multiple well-trained teacher networks to the student network, we propose a novel Graph-guided Semantically Discriminative Mapping (GSDM) loss, which utilizes graph-guided ablation analysis to produce a good visual explanation highlighting the important regions across modalities and concurrently preserving the interrelations of original data. Experimental results on Berkeley-MHAD, UTDMHAD and MMAct datasets well demonstrate the effectiveness of our proposed SAKDN for adaptive knowledge transfer from wearable-sensors modalities to vision-sensors modalities.

## Datasets
[UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)        
[Berkeley-MHAD](https://tele-immersion.citris-uc.org/berkeley_mhad/)           
[MMAct](https://mmact19.github.io/2019/)        

## Codes 
The code is available now!     
For example, in the UTD dataset:     
Step1: Extract the GAF images.     
Step2: Run 1_UTD_SemanticFusion.py to train the teacher model.     
Step3: Run 2_run_UTD_All_SemanticAblationFusion_distill.py to conduct knowledge distillation.     

```
@ARTICLE{yliuSAKDN, 
author={Yangliu, Keze Wang, Guanbin Li, Liang Lin}, 
journal={submitted to IEEE Transactions on Image Processing}, 
title={Semantics-aware Adaptive Knowledge Distillation Networks for Sensor-to-Vision Action Recognition}, 
year={2020}, 
volume={}, 
number={}, 
pages={}, 
doi={}, 
ISSN={}, 
month={},}
``` 
