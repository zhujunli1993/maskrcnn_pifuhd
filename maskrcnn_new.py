import sys
sys.path.insert(0, 'Mask_RCNN')
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-input", type=str)
parser.add_argument(
    "-output", type=str)
args = parser.parse_args()
image_path = args.input


CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
  NAME = "coco_inference"
  
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

  NUM_CLASSES = len(CLASS_NAMES)

model = mrcnn.model.MaskRCNN(mode="inference", 
              config=SimpleConfig(),
              model_dir=os.getcwd())

model.load_weights(filepath="mask_rcnn_coco.h5", 
          by_name=True)

for img_path in os.listdir(args.input):
  if img_path.endswith('.txt'):
    img_path = img_path.split('_')[0]+'.png'
    img_path = os.path.join(args.input,img_path)
    image = cv2.imread(img_path)
    print(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = model.detect([image], verbose=0)
    r = r[0]
    mrcnn.visualize.display_instances(image=image, 
    boxes=r['rois'], 
    masks=r['masks'], 
    class_ids=r['class_ids'], 
    class_names=CLASS_NAMES, 
    scores=r['scores'])

    img_masked = np.zeros([image.shape[0], image.shape[1], 3])

    for i in range(3): 
      img_masked[:,:,i] = np.multiply(image[:,:,i],r['masks'][:,:,0])
      idx_bg = (img_masked[:,:,i]==0)
      img_masked[idx_bg,i] = 255
    output_folder = args.output  
    output_image_path = os.path.join(output_folder, img_path.split('.')[0]+'_masked.png')
    cv2.imwrite(output_image_path, img_masked)
