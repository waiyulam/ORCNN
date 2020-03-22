# ORCNN in Detectron2 
**Learning to See the Invisible: End-to-End Trainable Amodal Instance
Segmentation**    
*Waiyu Lam*
*Instructor: [Yong Jae Lee](https://web.cs.ucdavis.edu/~yjlee/)* 

Occlusion-aware RCNN propose an all-in-one, end to end trainable multi-task
model for semantic segmentation that simultaneously predicts amodal masks,
visible masks, and occlusion masks for each object instance in an image in a
single forward pass. On the COCO amodal dataset, ORCNN outperforms the current
baseline for amodal segmentation by a large margin.     

The amodal mask is defined as the union of the visible mask and the invisible
occlusion mask of the object.    
Person:
<img src="https://github.com/waiyulam/ORCNN/blob/master/Results/amodal_mask/Person.png" alt="person" width="400"/>


Bench: 
<img src="https://github.com/waiyulam/ORCNN/blob/master/Results/amodal_mask/bench.png" alt="bench" width="400"/>

In this repository, we provide the code to train and evaluate ORCNN. We also
provide tools to visualize occlusion mask annotation and results.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [Getting Started ORCNN](https://github.com/waiyulam/ORCNN/blob/master/docs/tutorials/getting_started_orcnn.md)


## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for
download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing ORCNN

If you use Detectron2 in your research or wish to refer to the baseline results
published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX
entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
