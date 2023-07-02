from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2

def load_model():
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(f"Jalilov/doc-segment")
    return model, feature_extractor

def predict(model, feature_extractor, image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # convert to numpy array
    logits = logits.detach().numpy()
    return logits

def numpyToPil(image):
    return Image.fromarray(image)

def pilToNumpy(image):
    return np.array(image)

def resizeImage(image, size):
    return cv2.resize(image, size)
    