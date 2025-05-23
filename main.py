from preprocessing import VideoParser
from renderer import GestureRenderer
import argparse
import cv2
import numpy as np

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--action', type=str, default='parse_video')
    parse.add_argument('--video_path', type=str, default='')
    parse.add_argument('--video_id', type=str, default='')

    parse.add_argument('--gloss', type=str, default='action')
    parse.add_argument('--target_gloss', type=str, default='computer')

    args = parse.parse_args()
    action = args.action


    match action:
        # parses a single video, given argument video_path
        case 'parse_video':
            video_parser = VideoParser()
            landmark_dir = "./landmark_data/"
            video_path = args.video_path

            import glob
            import os
            import pandas as pd

            left_landmarks, right_landmarks = video_parser.parse_video(args.video_path, args.video_id, args.gloss)
            npz_filename = f"{args.video_id}.npz"
            npz_filepath = os.path.join(landmark_dir, npz_filename)
            np.savez_compressed(
                npz_filepath,
                left=np.array(left_landmarks, dtype=object),
                right=np.array(right_landmarks, dtype=object)
            )
            df = pd.read_csv("gloss_to_gesture_mapping_condensed.csv")
            new_index = len(df)
            df.loc[new_index] = {
                'id': args.video_id,
                'gloss': args.gloss,
                'landmark_file': npz_filename
            }

            df.to_csv("gloss_to_gesture_mapping_condensed.csv")

            # create a dataset using all videos in data folder (Note: takes ~ 5 hours to run this on all data)
        case 'create_dataset':
            video_parser = VideoParser()
            video_parser.create_dataset()
        case 'render_all':
            video_renderer = GestureRenderer()
            video_renderer.fetch_all_data()
        case 'render':
            renderer = GestureRenderer(dict_path="gloss_to_gesture_mapping_condensed.csv",
                                       landmark_dir="./landmark_data/")
            # get target gloss from argparse
            target_gloss = args.target_gloss
            frame_generator = renderer.render_gesture_from_gloss(target_gloss)

            frame_count = 0
            for frame_bytes in frame_generator:
                nparr = np.frombuffer(frame_bytes.split(b'\r\n\r\n')[1].rsplit(b'\r\n', 1)[0], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow(f'Rendering: {target_gloss}', img)
                frame_count += 1

            cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
