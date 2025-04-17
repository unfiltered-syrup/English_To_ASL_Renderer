import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 # used to created landmark normalized list
import json
import os
import glob
import pandas as pd

class GestureRenderer:
    def __init__(self, dict_path="gloss_to_gesture_mapping.csv", # Changed to CSV based on VideoParser output
                 landmark_dir="./landmark_data/",
                 canvas_width=640, canvas_height=520): # Define canvas size
        try:
            mapping_df = pd.read_csv(dict_path)
            # convert csv to python dictionary with index being gloss
            self.gloss_gesture_mapping = mapping_df.set_index('gloss').to_dict('index')
            print(f"Gloss mapping loaded.")
        except Exception as e:
            print(f"Error loading gloss mapping from {dict_path}: {e}")

        self.landmark_dir = landmark_dir
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Initialize MediaPipe drawing utls
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.drawing_spec_points = self.mp_drawing.DrawingSpec(color=(200, 200, 0), thickness=2, circle_radius=2)
        self.drawing_spec_lines = self.mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2, circle_radius=2)


    def numpy_to_normalized_landmark_list(self, frame_landmarks_np):
        # this function converts the saved numpy array, and turn it into mediapipe normalized landmarklist
        if frame_landmarks_np is None or not isinstance(frame_landmarks_np, np.ndarray):
            return None
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(frame_landmarks_np.shape[0]):
            landmark = landmark_list.landmark.add()
            landmark.x = frame_landmarks_np[i, 0]
            landmark.y = frame_landmarks_np[i, 1]
            landmark.z = frame_landmarks_np[i, 2]
        return landmark_list

    def fetch_data(self, filename):
        # using the filename obtained from mapping, get landmark data
        filepath = os.path.join(self.landmark_dir, filename)
        try:
            landmark_data = np.load(filepath, allow_pickle=True)
            # get both hands
            left_landmarks = landmark_data.get('left')
            right_landmarks = landmark_data.get('right')

            print(f"Fetched data from {filename}: {len(left_landmarks)} frames.")
            return left_landmarks, right_landmarks
        except Exception as e:
            print(f"Error loading landmarks from {filepath}: {e}")
            return None, None

    def render_gesture_from_gloss(self, gloss):
        # this is the generator used to yield the frames
        landmark_info = self.gloss_gesture_mapping[gloss]
        landmark_file = landmark_info.get('landmark_file')

        left_landmarks, right_landmarks = self.fetch_data(landmark_file)

        # Iterate through each frame's landmarks
        num_frames = len(left_landmarks)
        for i in range(num_frames):
            # Create a blank canvas
            canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

            # Get landmarks for the frame
            left_frame = left_landmarks[i]
            right_frame = right_landmarks[i]

            # Convert numpy arrays to MediaPipe NormalizedLandmarkList
            left_nll = self.numpy_to_normalized_landmark_list(left_frame)
            right_nll = self.numpy_to_normalized_landmark_list(right_frame)

            # draw the hand only if it exists in the frame
            if left_nll:
                self.mp_drawing.draw_landmarks(
                    image=canvas,
                    landmark_list=left_nll,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec_points,
                    connection_drawing_spec=self.drawing_spec_lines
                    )

            if right_nll:
                self.mp_drawing.draw_landmarks(
                    image=canvas,
                    landmark_list=right_nll,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec_points,
                    connection_drawing_spec=self.drawing_spec_lines
                    )

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', canvas)
            if not ret:
                print(f"Error encoding frame {i} for gloss {gloss}")
                continue

            # Yield the frame in JPEG format
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        print(f"Finished rendering gesture")


