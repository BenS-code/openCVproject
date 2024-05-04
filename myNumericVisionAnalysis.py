import tkinter as tk
import os
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model


class NumericRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numeric Recognizer")
        self.root.geometry("600x650")  # Set the window size to 800x600 pixels

        # Create Frames
        self.left_frame = tk.Frame(self.root, bd=5)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.right_frame = tk.Frame(self.root, bd=5)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Left Pane Buttons
        self.connect_btn = tk.Button(self.left_frame, text="Connect", command=self.connect_camera)
        self.connect_btn.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.display_video_btn = tk.Button(self.left_frame, text="Display Video", command=self.display_video)
        self.display_video_btn.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.capture_image_btn = tk.Button(self.left_frame, text="Capture Image", command=self.capture_image)
        self.capture_image_btn.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.analyze_btn = tk.Button(self.left_frame, text="Analyze", command=self.analyze_image)
        self.analyze_btn.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # Text Entry
        self.text_entry = tk.Entry(self.left_frame)
        self.text_entry.grid(row=4, column=0, padx=10, pady=(50, 5), sticky="ew")

        # Right Pane
        self.video_frame = tk.Label(self.right_frame, text="Video Frame")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.image_frame = tk.Label(self.right_frame, text="Image Frame")
        self.image_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # Configure row and column weights to stretch objects with window resizing
        # self.root.grid_rowconfigure(0, weight=1)
        # self.root.grid_rowconfigure(1, weight=1)
        # self.root.grid_columnconfigure(0, weight=1)
        # self.root.grid_columnconfigure(1, weight=1)

        # Video Capture object
        self.cap = None

        self.image_path = ''

    def connect_camera(self):
        if self.cap is None or not self.cap.isOpened():
            # Connect to default camera
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.connect_btn.config(text="Disconnect")
                print("Connected to default camera.")
            else:
                print("Failed to connect to default camera.")
        else:
            # Disconnect camera
            self.cap.release()
            self.connect_btn.config(text="Connect")
            print("Camera disconnected.")

    def display_video(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame to fit the video frame widget
                frame_resized = cv2.resize(frame_rgb, (400, 300))

                # Convert frame to ImageTk format
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update video frame with new frame
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

                # Schedule the next update
                self.root.after(10, self.display_video)
            else:
                print("Failed to read frame from camera.")
        else:
            print("Camera is not connected.")

    def capture_image(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Get current date and time
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Create directory if it doesn't exist
                save_dir = "saved_images"
                os.makedirs(save_dir, exist_ok=True)

                # Save image with timestamp
                image_name = f"captured_image_{current_time}.jpg"
                self.image_path = os.path.join(save_dir, image_name)
                cv2.imwrite(self.image_path, frame)

                # Convert frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame to fit the image frame widget
                frame_resized = cv2.resize(frame_rgb, (400, 300))

                # Convert frame to ImageTk format
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update image frame with the captured frame
                self.image_frame.imgtk = imgtk
                self.image_frame.configure(image=imgtk)

                print(f"Image captured and saved as {self.image_path}")
            else:
                print("Failed to read frame from camera.")
        else:
            print("Camera is not connected.")

    def analyze_image(self):
        if self.cap is not None and self.cap.isOpened():
            # Load the saved image
            frame = cv2.imread(self.image_path)

            # Preprocess the image for the CNN model
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 28, 28, 1))  # Reshape to match model input shape

            # Load pre-trained CNN model
            model = load_model("digit_recognition_model.h5")  # Change this to your model path

            # Predict digit using the loaded model
            prediction = model.predict(reshaped)
            digit = np.argmax(prediction)

            # Display the predicted digit in the text entry
            self.text_entry.delete(0, tk.END)  # Clear previous entry
            self.text_entry.insert(0, str(digit))

            # Display the predicted digit on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(digit), (10, 50), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit the image frame widget
            frame_resized = cv2.resize(frame_rgb, (400, 300))

            # Convert frame to ImageTk format
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update image frame with the captured frame
            self.image_frame.imgtk = imgtk
            self.image_frame.configure(image=imgtk)

            print(f"Predicted digit: {digit}")
        else:
            print("Camera is not connected.")


if __name__ == "__main__":
    root = tk.Tk()
    app = NumericRecognizerApp(root)
    root.mainloop()
