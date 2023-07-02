import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils import (
    load_model,
    predict,
    numpyToPIL,
    PILToNumpy,
    resizeImage
)
# load image with PIL Image from data/document.jpg
image = Image.open("data/document.jpg")

# show image
model, feature_extractor = load_model()
logits = predict(model, feature_extractor, image)

pred = logits[0][0]

size = (image.size[0], image.size[1])
pred = resizeImage(pred, size)

# save image with numpy array
plt.imsave("data/prediction.png", pred, cmap="gray")
