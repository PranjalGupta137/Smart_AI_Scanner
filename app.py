import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from deepface import DeepFace
import numpy as np

st.title("🚀 AI Smart Mood & Identity Scanner")

# Identity ke liye reference image
# Yaad se 'me.jpg' folder mein rakhna
REFERENCE_PATH = "me.jpg"

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            # 1. Identity aur Mood Analysis
            # 'enforce_detection' false rakhte hain taaki crash na ho
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            verify = DeepFace.verify(img, REFERENCE_PATH, enforce_detection=False)
            
            is_me = "Verified: Pranjal" if verify['verified'] else "Unknown User"

            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                mood = res['dominant_emotion']

                # UI Graphics
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{is_me} | {mood}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception as e:
            pass

        return img

# WebRTC Streamer - Yeh browser ka camera use karega
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)