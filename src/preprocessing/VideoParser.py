import mediapipe as mp
import cv2
import numpy as np
import glob

class VideoParser:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3):
        self.mp_hands = mp.solutions.hands
        # settings for mediapipe hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame = None
        self.target_height = 520

    def create_dataset(self):
        for filename in glob.glob('./data/*'):
            print(filename)
            self.parse_video(filename)


    def parse_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        success = True
        count = 0
        print(video_path)
        while True:
            success, self.frame = cap.read()

            if not success:
                break

            # resize frame
            height, width = self.frame.shape[:2]
            height_scale = self.target_height / height
            self.frame = cv2.resize(self.frame, (int(width * height_scale), self.target_height))

            # conversion for landmark prediction
            RGB_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            landmarks, palm_loc = self.parse_frame(RGB_frame)

            count += 1

            # show image
            cv2.imshow('Hand Tracking', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                break


    def parse_frame(self, RGB_frame):
        mp_hand_object = self.hands.process(RGB_frame)

        if mp_hand_object.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in mp_hand_object.multi_hand_landmarks[0].landmark])
            palm_loc = landmarks[0]
        else:
            return None, None

        for hand_landmarks in mp_hand_object.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return landmarks, palm_loc


