from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2

class SegmentDocument:
    def __init__(self, image):
        """
        This class for segment document image

        Args:
            image: PIL Image
        """
        self.image = image
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained(f"Jalilov/doc-segment")

    def predict(self):
        """
        Predict image segmentation 

        Returns:
            logits: numpy array
        """
        inputs = self.feature_extractor(images=self.image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # convert to numpy array
        logits = logits.detach().numpy()
        return logits
    
    def numpyToPil(self):
        """
        Convert numpy array to PIL Image
        """
        return Image.fromarray(self.image)
    
    def pilToNumpy(self):
        """
        Convert PIL Image to numpy array
        """
        return np.array(self.image)
    
    def resizeImage(self,pred, size):
        """
        Resize image

        Args:
            pred: numpy array
            size: tuple (width, height)

        Returns:
            numpy array: resized image
        """
        return cv2.resize(pred, size)
    
    def segmentImage(self):
        """
        get segmentation image
        """
        logits = self.predict()
        pred = logits[0][0]
        size = (self.image.size[0], self.image.size[1])
        pred = self.resizeImage(pred, size)
        return pred

    def float32ToUint8(self, image):
        """
        Convert float32 to uint8

        Args:
            image: numpy array

        Returns:
            numpy array: uint8 image
        """
        return np.clip(image, 0, 255).astype(np.uint8)


class GetDocument:
    def __init__(self, mask_img, image):
        """
        This class for get document from image and segment document image

        Args:
            mask_img: numpy array
            image: PIL Image
        """
        self.mask_img = mask_img
        self.image = image

    def __threshold(self):
        """
        Thresholding document image
        """
        ret, thresh = cv2.threshold(self.mask_img, 2, 255, 0)
        return thresh
    
    def __getContours(self, thresh):
        """
        Get contours from threshold image

        Args:
            thresh: numpy array

        Returns:
            contours: numpy array
        """
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def __getBiggestContour(self, contours):
        """
        Get the biggest area from contours

        Args:
            contours: numpy array
        
        Returns:
            biggest: numpy array
        """
        biggest = None
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest
    
    def getPoints(self):
        """
        Get points from the biggest contour for transform image

        Returns:
            pts1: numpy array
            pts2: numpy array
        """
        thresh = self.__threshold()
        contours = self.__getContours(thresh)
        biggest = self.__getBiggestContour(contours)
        # reshape biggest 4, 2
        biggest = biggest.reshape((4, 2))

        pt1, pt2, pt3, pt4 = tuple(biggest)
        pts1 = np.array([pt1, pt4, pt3, pt2], np.float32)

        size = self.mask_img.shape
        h, w = size[:2]

        pts1 = np.float32(pts1)
        pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])

       

        return pts1, pts2

    def transform(self, img, inv=False):
        """
        Transform the image to the desired shape

        Args:
            img (np.array): image to be transformed
            shape (tuple): desired shape
        Returns:
            dst (np.array): transformed image
        
        """
        pts1, pts2 = self.getPoints()

        h, w = img.shape[:2]

        if inv:
            M = cv2.getPerspectiveTransform(pts2, pts1)
        else:
            M = cv2.getPerspectiveTransform(pts1, pts2)
        
        dst = cv2.warpPerspective(img, M, (w, h))

        return dst