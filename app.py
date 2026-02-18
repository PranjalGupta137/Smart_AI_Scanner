import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
from deepface import DeepFace
import os

# Page ki basic settings
st.set_page_config(page_title="AI Live Mood Scanner", layout="centered")
st.title("🤖 Live Mood & Identity Tracker")

# 1. Identity File ki Location (Check karna ki GitHub par me.png isi naam se ho)
REFERENCE_PATH = "me.png" 

if not os.path.exists(REFERENCE_PATH):
    st.error(f"Error: '{REFERENCE_PATH}' file nahi mili! GitHub par apni photo upload karo jiska naam 'me.png' ho.")
    st.stop()

# 2. STUN Server Configuration (Connection error hatane ke liye)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 3. AI Processor Class: Jo har second frame ko check karegi
class MoodProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Frame ko array (image) format mein convert karna
        img = frame.to_ndarray(format="bgr24")

        try:
            # Identity Check: Kya saamne Pranjal hai?
            # detector_backend='opencv' use kiya hai taaki speed fast rahe
            verify = DeepFace.verify(img, REFERENCE_PATH, enforce_detection=False, detector_backend='opencv')
            is_me = "Pranjal (Verified)" if verify['verified'] else "Unknown User"
            
            # Mood Analysis: Happy, Sad, Angry, etc.
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                mood = res['dominant_emotion']

                # Green box agar aap ho, Red box agar unknown
                color = (0, 255, 0) if verify['verified'] else (0, 0, 255)
                
                # Drawing Box and Text
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                cv2.putText(img, f"{is_me}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f"Mood: {mood}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        except Exception as e:
            # Agar face detect nahi hua toh error nahi aayega, bas blank frame chalega
            pass

        return img

# 4. Web Interface (Streamer)
st.write("Niche 'START' button par click karein aur camera access allow karein:")

webrtc_streamer(
    key="mood-scanner",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=MoodProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Smooth performance ke liye
)

st.divider()
st.info("Tip: Per second accuracy ke liye chehre par acchi light rakhein.")
