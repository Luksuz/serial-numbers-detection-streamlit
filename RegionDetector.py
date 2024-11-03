import torch
import streamlit as st

class RegionDetector:
    weights_path = "./weights/best_region.pt"
    
    def __init__(self):
        with st.spinner("Loading Region Detection Model..."):
            self.model = torch.hub.load('./yolov5', 'custom', path=RegionDetector.weights_path, source="local")
    
    def _get_predictions(self, img, conf=0.4):
        self.model.conf = conf
        with torch.no_grad():
            preds = self.model(img)
        labels = preds.xyxy[0]
        return labels, img
    
    def get_serial_region(self, img):
        labels, img = self._get_predictions(img)
        if len(labels) == 0:
            return None, None
        
        cropped_images = []
        for label in sorted(labels, key=lambda x: x[5], reverse=True)[:2]:
            x1, y1, x2, y2 = [ten.item() for ten in label[:4]]
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_img)
        
        best_label = max(labels, key=lambda x: x[5])
        x1, y1, x2, y2 = best_label[0].item(), best_label[1].item(), best_label[2].item(), best_label[3].item()
        detected = img.crop((x1, y1, x2, y2))
        return detected, cropped_images
