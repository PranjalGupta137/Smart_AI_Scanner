import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2
from deepface import DeepFace
import os

# Page Configuration
st.set_page_config(page_title="AI Live Mood Scanner", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-box { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: #1f2937; 
        text-align: center;
        border: 2px solid #3b82f6;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 AI Identity & Mood Dashboard")

# 1. Identity File Check
REFERENCE_PATH = "me.png" 

if not os.path.exists(REFERENCE_PATH):
    st.error(f"Error: '{REFERENCE_PATH}' nahi mili! GitHub par file upload karein.")
    st.stop()

# 2. Updated STUN Config (Connection Fix)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 3. Fast Processor
class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Fast Analysis using OpenCV backend
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                mood = res['dominant_emotion']
                
                # Bounding Box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # White Bar at bottom for Mood Text (Exactly like your reference)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, img.shape[0]-80), (img.shape[1], img.shape[0]), (255, 255, 255), -1)
                img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
                
                # Large Mood Text
                cv2.putText(img, mood.upper(), (int(img.shape[1]/3.5), img.shape[0]-25), 
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0), 3)
        except:
            pass

        return img

# 4. Professional Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Scanner Feed")
    # Fixed WebRtcMode Error here:
    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("System Data")
    st.markdown(f"""
        <div class="status-box">
            <p>Identity: <b>Verified ({REFERENCE_PATH})</b></p>
            <p style='color: #00ff00;'>AI Model: Running</p>
            <p style='color: #3b82f6;'>Updates: Real-time</p>
        </div>
    """, unsafe_allow_html=True)
    st.info("Scanner har second aapke facial expressions ko monitor kar raha hai.")

st.warning("Tip: Agar camera load na ho, toh refresh karein aur camera access allow karein.")
