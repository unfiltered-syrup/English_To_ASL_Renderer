import numpy as np
import glob

class GestureRenderer:
    def __init__(self):
        pass

    def fetch_data(self, filename, data_path="./landmark_data/"):
        # read landmark data from npz files
        landmark_data = np.load(filename, allow_pickle=True)
        left_landmarks = landmark_data['left']
        right_landmarks = landmark_data['right']
        print(left_landmarks)
        print(right_landmarks)
        print("*" * 30)
        return left_landmarks, right_landmarks

    def fetch_all_data(self):
        for filename in glob.glob('./landmark_data/*.npz'):
            left_landmarks, right_landmarks = self.fetch_data(filename)
            self.render_hands(left_landmarks, right_landmarks)

    # TODO: Render the fetched data on screen
    def render_hands(self, left_landmarks, right_landmarks):
        self.render_left()
        self.render_right()

    def render_left(self):
        pass

    def render_right(self):
        pass