import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2
from deepface import DeepFace
import os

# Page configuration
st.set_page_config(page_title="AI Emotion Dashboard", layout="wide")

# Custom CSS for that big bold look you wanted
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

st.title("🎭 Real-Time Emotion Detection")

# 1. Identity File Check
REFERENCE_PATH = "me.png" 

if not os.path.exists(REFERENCE_PATH):
    st.error(f"Error: '{REFERENCE_PATH}' file nahi mili! GitHub par 'me.png' upload karein.")
    st.stop()

# 2. STUN Configuration (Orange Box fix)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 3. AI Processing Logic
class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Identity and Emotion analysis
            # detector_backend='opencv' is the fastest for web servers
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            verify = DeepFace.verify(img, REFERENCE_PATH, enforce_detection=False, detector_backend='opencv')
            
            is_me = "Pranjal (Verified)" if verify['verified'] else "Unknown User"
            
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                mood = res['dominant_emotion']
                
                # Professional Bounding Box
                color = (0, 255, 0) if verify['verified'] else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Bottom White Bar (Like your reference image)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, img.shape[0]-90), (img.shape[1], img.shape[0]), (255, 255, 255), -1)
                img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
                
                # Large Bold Text at the bottom
                cv2.putText(img, mood.upper(), (int(img.shape[1]/3.5), img.shape[0]-35), 
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0), 3)
        except:
            pass

        return img

# 4. Streamlit UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Scanner Feed")
    # Fixed the AttributeError by using the imported WebRtcMode directly
    webrtc_streamer(
        key="ai-mood-scanner-final",
        mode=WebRtcMode.SENDRECV, # Yeh line ab error nahi degi
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Biometric Info")
    st.markdown(f"""
        <div class="status-box">
            <p>User: <b>Pranjal</b></p>
            <p>Status: <span style='color:#00ff00;'>ONLINE</span></p>
            <p>Device: Web Camera</p>
        </div>
    """, unsafe_allow_html=True)
    st.info("System har frame ko scan karke aapki emotions detect kar raha hai.")

st.warning("Tip: Agar screen black dikhe, toh 'START' button dabakar browser permission check karein.")
