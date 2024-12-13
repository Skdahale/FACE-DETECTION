import cv2
import sys

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

def detect_faces(grayscale, image, cascades, is_webcam):
    """Detect faces in an image and draw rectangles around them."""
    frontal_cascade, profile_cascade = cascades
    # Detect frontal faces
    frontal_faces = frontal_cascade.detectMultiScale(
        grayscale, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
    )
    image = draw_rectangle(image, (247, 173, 60), frontal_faces)

    if not is_webcam:
        # Detect profile faces
        profile_faces = profile_cascade.detectMultiScale(
            grayscale, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
        )
        flipped_grayscale = cv2.flip(grayscale, 1)
        profile_faces_flipped = profile_cascade.detectMultiScale(
            flipped_grayscale, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
        )

        image = draw_rectangle(image, (120, 217, 30), profile_faces)
        image = cv2.flip(image, 1)
        image = draw_rectangle(image, (120, 217, 30), profile_faces_flipped)
        image = cv2.flip(image, 1)

    return image

def process_webcam(cascades):
    """Use webcam for real-time face detection."""
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error accessing webcam.")
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = detect_faces(grayscale, frame, cascades, is_webcam=True)
        frame = cv2.flip(frame, 1)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def process_image(image_path, cascades):
    """Use an image file for face detection."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = detect_faces(grayscale, image, cascades, is_webcam=False)

    cv2.imshow("Face Detection", image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def main():
    cascades = load_cascades()

    if len(sys.argv) == 1:
        process_webcam(cascades)
    elif len(sys.argv) == 2:
        process_image(sys.argv[1], cascades)
    else:
        print("Usage: python face-detect-haar.py [optional.jpg]")
        sys.exit(1)

if __name__ == "__main__":
    main()
