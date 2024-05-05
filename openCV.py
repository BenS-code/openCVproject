import cv2
import numpy as np


def detect_and_draw_digits(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Thresholding to binarize the image
    _, thresholded = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the segments for each digit (0-9)
    segments = {
        0: [1, 1, 1, 1, 1, 1, 0],
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1]
    }

    # Iterate through contours to find potential digit regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Adjust the criteria for considering a region as a potential digit
        if 0.9 <= aspect_ratio <= 1.1 and 20 < w < 200 and 20 < h < 200:
            roi = thresholded[y:y + h, x:x + w]
            digit_segments = segment_detection(roi)

            # Compare detected segments with predefined segments for each digit
            digit = None
            for key, value in segments.items():
                if digit_segments == value:
                    digit = key
                    break

            # If a digit is recognized, draw it on the frame
            if digit is not None:
                cv2.putText(frame, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def segment_detection(roi):
    # Define segment positions
    segments = [
        (3, 0, 4, 0),  # Top
        (4, 1, 4, 2),  # Top-right
        (4, 3, 4, 4),  # Bottom-right
        (3, 5, 4, 5),  # Bottom
        (2, 3, 2, 4),  # Bottom-left
        (2, 1, 2, 2),  # Top-left
        (3, 2, 3, 3)  # Middle
    ]

    # Iterate through segments and check if they are present in the ROI
    digit_segments = []
    for segment in segments:
        pt1, pt2 = segment[:2], segment[2:]
        x1, y1 = pt1
        x2, y2 = pt2
        if np.any(roi[y1:y2 + 1, x1:x2 + 1] > 128):  # Check if any pixel in the segment is white
            digit_segments.append(1)
        else:
            digit_segments.append(0)
    return digit_segments


def main():
    # Connect to the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to connect to camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect and draw potential digits on the frame
        detect_and_draw_digits(frame)

        # Display the resulting frame
        cv2.imshow('7-Segment Digit Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
