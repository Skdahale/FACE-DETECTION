import numpy as np
import sys
import cv2
from imutils.video import VideoStream # type: ignore
import imutils # type: ignore
import time

def load_dnn_model(prototxt_path, caffemodel_path):
    """Load DNN model from prototxt and caffemodel files."""
    return cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def draw_rectangle(image, color, box, thickness=2):
    """Draw a rectangle with a top bar above the detected face."""
    (x, y, x1, y1) = box
    h = y1 - y
    bar_length = int(h / 8)
    cv2.rectangle(image, (x, y - bar_length), (x1, y), color, -1)
    cv2.rectangle(image, (x, y - bar_length), (x1, y), color, thickness)
    cv2.rectangle(image, (x, y), (x1, y1), color, thickness)
    return image

def change_font_scale(box_height, base_scale=0.4):
    """Adjust font scale dynamically based on the box height."""
    base_height = 108
    return (box_height / base_height) * base_scale

def detect_faces_dnn(image, net, conf_threshold=0.3, mean_values=(104.0, 177.0, 124.0)):
    """Detect faces in an image using a DNN model and draw rectangles."""
    h, w = image.shape[:2]
    resized_image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), mean_values)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")
            font_scale = change_font_scale(box[3] - box[1])
            image = draw_rectangle(image, (247, 173, 62), box)

            text = f"{confidence * 100:.2f}%"
            text_y = max(box[1] - 10, 10)
            cv2.putText(image, text, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return image

def process_webcam(net):
    """Use webcam for real-time face detection."""
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = detect_faces_dnn(frame, net)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

def process_image(image_path, net):
    """Detect faces in a static image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    image = detect_faces_dnn(image, net)
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    prototxt_path = "deploy.prototxt.txt"
    caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"

    net = load_dnn_model(prototxt_path, caffemodel_path)

    if len(sys.argv) == 1:
        process_webcam(net)
    elif len(sys.argv) == 2:
        process_image(sys.argv[1], net)
    else:
        print("Usage: python face-detect-dnn.py [optional.jpg]")
        sys.exit(1)

if __name__ == "__main__":
    main()
