import numpy as np
import glob
import json

class GestureRenderer:
    def __init__(self, dict_path = "gloss_to_gesture_mapping.json"):
        # loads json data of gloss and gesture mapping
        self.gloss_gesture_mapping = json.load(open(dict_path))

    def fetch_data(self, filename, data_path="./landmark_data/"):
        # read landmark data from npz files
        landmark_data = np.load(data_path+filename, allow_pickle=True)
        left_landmarks = landmark_data['left']
        right_landmarks = landmark_data['right']
        '''
        print(left_landmarks)
        print(right_landmarks)
        print("*" * 30)
        '''
        return left_landmarks, right_landmarks

    def render_gesture_from_gloss(self, gloss):
        landmark_path = self.gloss_gesture_mapping[gloss]['landmark']
        backup_path = self.gloss_gesture_mapping[gloss]['backup_landmark']
        if landmark_path is not None:
            self.fetch_data(landmark_path)
        else:
            self.fetch_data(backup_path)


    def fetch_all_data(self):
        for filename in glob.glob('./landmark_data/*.npz'):
            left_landmarks, right_landmarks = self.fetch_data(filename)
            self.render_hands(left_landmarks, right_landmarks)

    # TODO: Render the fetched data on screen
    def render_hands(self, left_landmarks, right_landmarks):
        self.render_left(left_landmarks)
        self.render_right(right_landmarks)

    def render_left(self, left_landmarks):
        pass

    def render_right(self, right_landmarks):
        pass