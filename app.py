import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def load_cascades():
    """Load Haar cascade classifiers for frontal and profile faces."""
    frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    return frontal_face, profile_face

def draw_rectangle(image, color, faces, thickness=2):
    """Draw rectangles around detected faces."""
    for (x, y, w, h) in faces:
        bar_length = int(h / 8)
        bar_width = w
        # Draw top bar and surrounding rectangle
        cv2.rectangle(image, (x, y - bar_length), (x + bar_width, y), color, -1)
        cv2.rectangle(image, (x, y - bar_length), (x + bar_width, y), color, thickness)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.cascades = load_cascades()

    def transform(self, frame):
        """Detect faces and apply bounding boxes."""
        frontal_cascade, profile_cascade = self.cascades
        img = frame.to_ndarray(format="bgr24")
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect frontal faces
        frontal_faces = frontal_cascade.detectMultiScale(
            grayscale, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
        )
        img = draw_rectangle(img, (247, 173, 60), frontal_faces)

        # Detect profile faces (optional)
        profile_faces = profile_cascade.detectMultiScale(
            grayscale, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
        )
        img = draw_rectangle(img, (120, 217, 30), profile_faces)

        return img

def main():
    st.title("Face Detection with Camera Access")
    st.write("This app requires access to your camera. Please grant permissions when prompted.")

    # Use streamlit-webrtc for camera access
    webrtc_streamer(
        key="face-detection",
        video_transformer_factory=FaceDetectionTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Video only
    )

if __name__ == "__main__":
    main()
