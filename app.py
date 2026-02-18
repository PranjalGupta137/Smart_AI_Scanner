import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
from deepface import DeepFace
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="AI Smart Scanner", layout="centered")
st.title("🤖 AI Identity & Mood Scanner")

# 1. Reference Image Check (Identity ke liye)
REFERENCE_PATH = "me.png"

if not os.path.exists(REFERENCE_PATH):
    st.error(f"Error: '{REFERENCE_PATH}' file nahi mili! GitHub par apni photo upload karo.")
    st.stop()

# 2. WebRTC Configuration (Cloud Deployment ke liye zaroori)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 3. AI Processing Logic
class AIProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Frame ko array mein convert karein
        img = frame.to_ndarray(format="bgr24")

        try:
            # Identity Verification (Kya ye Pranjal hai?)
            # Enforce_detection=False taaki agar face door ho toh crash na ho
            verify = DeepFace.verify(img, REFERENCE_PATH, enforce_detection=False, detector_backend='opencv')
            is_me = "Verified: Pranjal" if verify['verified'] else "Unknown User"
            color = (0, 255, 0) if verify['verified'] else (0, 0, 255)

            # Mood Analysis
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                mood = res['dominant_emotion']

                # UI Graphics drawing
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{is_me}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(img, f"Mood: {mood}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        except Exception as e:
            # Agar face detect nahi hua toh normal frame dikhao
            pass

        return img

# 4. Streamlit UI
st.write("Niche diye gaye button se camera start karein:")

webrtc_streamer(
    key="ai-scanner",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=AIProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("Tip: Acchi accuracy ke liye light mein rahein aur front face dikhayein.")

