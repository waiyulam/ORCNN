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
### Inference with Pre-trained Models
See [Getting Started Amodal](https://github.com/waiyulam/ORCNN/blob/master/Amodal_demo.ipynb)
### Training & Evaluation & Visualization
See [Getting Started ORCNN](https://github.com/waiyulam/ORCNN/blob/master/ORCNN%20Training.ipynb)

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing ORCNN

```BibTeX
@article{DBLP:journals/corr/abs-1804-08864,
  author    = {Patrick Follmann and
               Rebecca K{\"{o}}nig and
               Philipp H{\"{a}}rtinger and
               Michael Klostermann},
  title     = {Learning to See the Invisible: End-to-End Trainable Amodal Instance
               Segmentation},
  journal   = {CoRR},
  volume    = {abs/1804.08864},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.08864},
  archivePrefix = {arXiv},
  eprint    = {1804.08864},
  timestamp = {Mon, 13 Aug 2018 16:46:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1804-08864.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
