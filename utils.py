from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2

class SegmentDocument:
    def __init__(self, image):
        self.image = image
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained(f"Jalilov/doc-segment")

    def predict(self):
        inputs = self.feature_extractor(images=self.image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # convert to numpy array
        logits = logits.detach().numpy()
        return logits
    
    def numpyToPil(self):
        return Image.fromarray(self.image)
    
    def pilToNumpy(self):
        return np.array(self.image)
    
    def resizeImage(self,pred, size):
        return cv2.resize(pred, size)
    
    def segmentImage(self):
        logits = self.predict()
        pred = logits[0][0]
        size = (self.image.size[0], self.image.size[1])
        pred = self.resizeImage(pred, size)
        return pred