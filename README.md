This is my own re-implementation model from ORCNN and only used for baseline model in our researching. The model is built based on the Dectectron2 and there is no official code that was used to obtain the results of the paper. Please reached the contact below if there is any concerns about the source code:<br>

Email: waiyu0616@gmail.com 

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

```
@inproceedings{follmann2019learning,

  author    = {Patrick Follmann and

               Rebecca K{\"{o}}nig and

               Philipp H{\"{a}}rtinger and

               Michael Klostermann and

               Tobias B{\"{o}}ttger},

  title     = {Learning to See the Invisible: End-to-End Trainable Amodal Instance

               Segmentation},

  booktitle = {{IEEE} Winter Conference on Applications of Computer Vision, {WACV}

               2019, Waikoloa Village, HI, USA, January 7-11, 2019},

  pages     = {1328--1336},

  publisher = {{IEEE}},

  year      = {2019},

  url       = {https://doi.org/10.1109/WACV.2019.00146},

  doi       = {10.1109/WACV.2019.00146},
}
```


