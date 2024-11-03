import streamlit as st
from PIL import Image, ImageDraw
from roboflow import Roboflow
import os


class CharactersDetector:
    digit_to_char = {
        "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S",
        "6": "G", "7": "T", "8": "B", "9": "O"
    }
    char_to_digit = {
        "A": "4", "B": "8", "C": "0", "D": "0", "E": "3", "F": "3",
        "G": "6", "H": "0", "I": "1", "J": "1", "K": "0", "L": "1",
        "M": "0", "N": "7", "O": "0", "P": "9", "Q": "0", "R": "0",
        "S": "5", "T": "1", "U": "0", "V": "0", "W": "0", "X": "0",
        "Y": "1", "Z": "2"
    }
    
    def __init__(self):
        with st.spinner("Loading Characters Detection Model..."):
            self.model = self._load_model()
    
    def _load_model(self):
        rf = Roboflow(api_key=st.secrets.get("ROBOFLOW_API_KEY"))
        project = rf.workspace().project("container-characters-detection")
        model = project.version(4).model
        return model

    def _get_predictions(self, img_path, conf=0.4):
        # Roboflow's predict method expects a file path
        labels = self.model.predict(img_path, confidence=conf, overlap=30).json()["predictions"]
        img = Image.open(img_path)
        return labels, img

    def draw_bbs(self, labels, img):        
        draw = ImageDraw.Draw(img)
        for label in labels:
            x, y = label["x"], label["y"]
            w, h = label["width"] / 2, label["height"] / 2
            x1, y1, x2, y2 = x - w, y - h, x + w, y + h
            draw.rectangle((x1, y1, x2, y2), outline="green", width=2)
        return img  # Return the image with bounding boxes

    def sort_and_read_characters(self, img_path, conf=0.4, show=False):
        with st.spinner("Detecting Characters..."):
            labels, img = self._get_predictions(img_path, conf)
        
        if not labels:
            return "No characters detected."

        # Sort labels based on x-coordinate (left to right)
        sorted_labels = sorted(labels, key=lambda x: x["x"])
        chars = [str(pred["class"]) for pred in sorted_labels[:11]]

        # Map digits and characters using homoglyphs
        for i in range(len(chars[:4])):
            chars[i] = CharactersDetector.digit_to_char.get(chars[i], chars[i])
        for i in range(4, len(chars[:11])):
            chars[i] = CharactersDetector.char_to_digit.get(chars[i], chars[i])

        formatted_chars = f"{''.join(chars[:4])}-{''.join(chars[4:-1])}-{chars[-1]}"

        if show:
            img_with_bbs = self.draw_bbs(labels, img)
            return formatted_chars, img_with_bbs

        return formatted_chars