import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils import SegmentDocument, GetDocument
# load image with PIL Image from data/document.jpg
image = Image.open("data/doc1.jpg")

segment = SegmentDocument(image)

# segmentation document image
doc = segment.segmentImage()

# float32 to uint8
doc = segment.float32ToUint8(doc)


get_doc = GetDocument(doc, image)

# image convert PIL to numpy
image = np.array(image)

transdormed = get_doc.transform(image)

plt.imshow(doc)
plt.show()