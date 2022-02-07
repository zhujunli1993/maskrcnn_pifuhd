import cv2
import argparse

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-video", type=str)
args = parser.parse_args()

filename = args.video
vidcap = cv2.VideoCapture(filename)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("pifuhd/sample_images/frame%d.png" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
