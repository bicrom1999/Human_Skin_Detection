import cv2
import numpy as np

def detect_skin(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for the skin color range
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply the binary mask to the original image
    skin_image = cv2.bitwise_and(image, image, mask=mask)

    return skin_image

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    skin_image = detect_skin(frame)
    cv2.imshow('Original', frame)
    cv2.imshow('Skin', skin_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()