import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
from deepface import DeepFace
import os
import numpy as np

# Page configuration for a professional look
st.set_page_config(page_title="AI Emotion Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-box { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: #1f2937; 
        text-align: center;
        border: 2px solid #3b82f6;
    }
    .mood-text { 
        font-size: 50px !important; 
        font-weight: bold; 
        text-transform: uppercase;
        color: #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Real-Time Emotion Detection Dashboard")

# 1. Identity Path Fix
REFERENCE_PATH = "me.png" 

# 2. STUN Configuration to fix the "Orange Box" connection error
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Identity and Emotion analysis
            # Using 'opencv' backend for maximum speed per second
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                mood = res['dominant_emotion']
                
                # Draw professional bounding box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add a semi-transparent overlay for the label (jaisa image mein tha)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, img.shape[0]-100), (img.shape[1], img.shape[0]), (255, 255, 255), -1)
                alpha = 0.8 
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                
                # Big Bold Mood Text on bottom (Exactly like your reference image)
                cv2.putText(img, mood.upper(), (int(img.shape[1]/3), img.shape[0]-40), 
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 0), 3)
        except Exception as e:
            pass

        return img

# Layout: UI ko professional banane ke liye columns use kiye hain
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video Feed")
    webrtc_streamer(
        key="emotion-detection",
        mode=st.WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("System Status")
    st.markdown(f"""
        <div class="status-box">
            <p style='color: white;'>Identity File: <b>{REFERENCE_PATH}</b></p>
            <p style='color: #00ff00;'>AI Model: Active</p>
            <p style='color: #3b82f6;'>Backend: TensorFlow + DeepFace</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("Scanner har second aapke facial expressions ko analyze kar raha hai.")

st.warning("Note: Agar camera load na ho, toh browser permissions check karein aur refresh karein.")
