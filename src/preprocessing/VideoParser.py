import mediapipe as mp
import cv2
import numpy as np
import glob
import json
import os
import pandas as pd


class VideoParser:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, debug_mode=False):
        self.mp_hands = mp.solutions.hands
        # settings for mediapipe hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        # drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame = None
        self.target_height = 520
        self.debug_mode = debug_mode

    def create_dataset(self, index_file="./src/utilities/WLASL_v0.3.json", landmark_dir="./landmark_data"):
        content = json.load(open(index_file))
        metadata_list = []
        i = 0
        for entry in content:
            gloss = entry['gloss']
            instances = entry['instances']
            for inst in instances:
                video_id = inst['video_id']
                video_path = "./raw_videos/" + video_id + ".mp4"
                if glob.glob(video_path):
                    print(f"video {video_id} found for {gloss}")
                    left_landmarks, right_landmarks = self.parse_video(video_path, video_id, gloss)
                    npz_filename = f"{video_id}.npz"
                    npz_filepath = os.path.join(landmark_dir, npz_filename)
                    try:
                        np.savez_compressed(
                            npz_filepath,
                            left=np.array(left_landmarks, dtype=object),
                            right=np.array(right_landmarks, dtype=object)
                        )

                        metadata_list.append({
                            'id': video_id,
                            'gloss': gloss,
                            'landmark_file': npz_filename
                        })
                    except Exception as e:
                        print(f"Error saving landmarks for {video_id}: {e}")
                    i += 1


        if metadata_list:
            dataset = pd.DataFrame(metadata_list)
            csv_output_path = "gloss_to_gesture_mapping.csv"
            dataset.to_csv(csv_output_path, index=False)
            print(f"Metadata saved to {csv_output_path}")

    def parse_video(self, video_path, video_id, gloss):
        cap = cv2.VideoCapture(video_path)
        success = True
        count = 0
        left_landmarks = []
        right_landmarks = []

        while True:
            #print(f"------------------------Frame {count}---------------------------------")
            success, self.frame = cap.read()

            if not success:
                break

            # resize frame
            height, width = self.frame.shape[:2]
            height_scale = self.target_height / height
            self.frame = cv2.resize(self.frame, (int(width * height_scale), self.target_height))

            # conversion for landmark prediction
            RGB_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            left_hand_landmarks, left_palm_loc, right_hand_landmarks, right_palm_loc  = self.parse_frame(RGB_frame)

            #print(left_hand_landmarks)
            #print(right_hand_landmarks)
            left_landmarks.append(left_hand_landmarks)
            right_landmarks.append(right_hand_landmarks)

            count += 1

            # show image if debugging
            if self.debug_mode:
                cv2.imshow('Hand Tracking', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return left_landmarks, right_landmarks


    def parse_frame(self, RGB_frame):
        mp_hand_object = self.hands.process(RGB_frame)

        landmarks_left, palm_loc_left, landmarks_right, palm_loc_right = None, None, None, None

        if mp_hand_object.multi_hand_landmarks:
            for i in range(len(mp_hand_object.multi_hand_landmarks)):
                hand_landmarks_proto = mp_hand_object.multi_hand_landmarks[i]
                handedness_proto = mp_hand_object.multi_handedness[i]
                label = handedness_proto.classification[0].label
                landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_proto.landmark])

                if label == "Left":
                    landmarks_left = landmarks_np
                    palm_loc_left = landmarks_np[0]
                elif label == "Right":
                    landmarks_right = landmarks_np
                    palm_loc_right = landmarks_np[0]
        else:
            return None, None, None, None

        for hand_landmarks in mp_hand_object.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return landmarks_left, palm_loc_left, landmarks_right, palm_loc_right


