# Data loading 
- dataset_dicts (list) is a list of annotation data registered from the dataset.
- DatasetFromList (data.Dataset) takes a dataset_dicts and wrap it as a torch dataset.
- MapDataset (data.Dataset) calls DatasetMapper class to map each element of DatasetFromList. It loads images, transforms images and annotations, and converts annotations to an ‘Instances’ object.
# Loading annotation data
from detectron2.data import DatasetCatalog
from mydataset import load_mydataset_json
def register_mydataset_instances(name, json_file):
    DatasetCatalog.register(name, lambda: load_mydataset_json(json_file, name))

# Mapping data
During training, registered annotation records are picked one by one. We need actual image data (not path) and corresponding annotations. The dataset mapper (DatasetMapper) deals with the records to add an ‘image’ and ‘Instances’ to the dataset_dict. ‘Instances’ are the ground truth structure object of Detectron 2.

1. Load and transform images
An image specified by ‘file name’ is loaded by read_image function. Loaded image is transformed by pre-defined transformers (such as left-right flip) and finally the image tensor whose shape is (channel, height, width) is registered.
2. Transform annotations
The ‘annotations’ of dataset_dict are transformed by the transformations performed on the images. For example, if the image has been flipped, the box coordinates are changed to the flipped location.
3. Convert annotations to Instances
The annotations are converted to Instances by this function called in the dataset mapper. ‘bbox’ annotations are registered to Boxes structure object which can store a list of bounding boxes. ‘category_id’ annotations are simply convereted to a torch tensor.

After mapping, the dataset_dict looks like:
{'file_name': 'imagedata_1.jpg',
'height': 640, 'width': 640, 'image_id': 0,
'image': tensor([[[255., 255., 255.,  ...,  29.,  34.,  36.],...[169., 163., 162.,  ...,  44.,  44.,  45.]]]),
'instances': {
'gt_boxes': Boxes(tensor([[ 100.58, 180.66, 214.78, 283.95],
[180.58, 162.66, 204.78, 180.95]])),
'gt_classes': tensor([9, 9]),
}

# Tensorbard kill 
I solved this problem by this way - (actually in my ssh, sometimes CTRL+C don't work properly. Then I use this)

Get the running tensorboard process details

ps -ef|grep tensorboard

Sample Output: uzzal_x+  4585  4413  0 02:46 pts/4    00:00:01  bin/python /bin/tensorboard --logdir=runs/

Kill the process using pid (process id)

kill -9 <pid>

first number 4585 is my current pid for tensorflow

# data loading 
 detection_utils.py : update the transformation and anno loading function if adding any new data 