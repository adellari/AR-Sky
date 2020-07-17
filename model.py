import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models
from torchvision.models.resnet import model_urls

from PIL import Image
from PIL import ImageFile

import numpy as np
import io
import os
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

#flag describing if gpu has cuda cores https://pytorch.org/docs/stable/cuda.html
useCuda = torch.cuda.is_available()

#we don't care about image metadata
ImageFile.LOAD_TRUNCATED_IMAGES = True

#initializes our prediction model
def get_instance_segmentation_model(num_classes=2):
    model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    #model = torch.load('./models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth')
    #which features do we want our model to look for
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    #only put a mask around the sky features we're looking for
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    #our model was trained with 256 layers, so we init with the same amount
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transforms():
    return T.Compose([T.ToTensor()])

def transform_image(img_bytes):
    img_transforms = get_transforms()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img_transforms(img).unsqueeze(0)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr>=0.5
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

#load our model and initialize it
model_path = os.path.join("models", "sky-region_mask_r-cnn_resnet50-fpn-1579167716")
model = get_instance_segmentation_model()

#specify where our model is expected to run
device = 'cuda: 0' if useCuda else 'cpu'
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

#move our model to the gpu if we have one
if useCuda: 
    model.cuda()
    
model.eval()

#runs our prediction model on an image input and outputs a mask
def get_mask_image(img_bytes):
    
    #Use our transforms on the input image (make it grayscale, downsample 
    #since less pixels = faster compute time) 
    
    img = transform_image(img_bytes)
    
    img = img.cuda() if useCuda else img   #use the "cuda()" to utilize gpu if it has cuda cores
    
    #print("--- %s seconds --- | utilizing cuda: %r" % (time.time() - start_time, useCuda))
    
    #run our prediction model with no gradients
    with torch.no_grad():
        prediction = model(img)
    
   #get the best prediction output mask
    mask = prediction[0]['masks'][0, 0]
    
    #move our output ot the cpu in case it ran on gpu, then convert the output array to image
    mask = Image.fromarray(mask.cpu().mul(255).byte().numpy())
    
    #invert colors (this can come in handy based on how we want to juxtapose our sky images)
    #mask = 1 - mask
    #print("--- %s seconds --- | utilizing cuda: %r" % (time.time() - start_time, useCuda))
    return mask

