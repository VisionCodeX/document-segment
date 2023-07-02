import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils import SegmentDocument
# load image with PIL Image from data/document.jpg
image = Image.open("data/doc1.jpg")

segment = SegmentDocument(image)

doc = segment.segmentImage()
# show image
plt.imshow(doc, cmap="gray")
plt.show()
