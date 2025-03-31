from preprocessing import VideoParser
from renderer import GestureRenderer
import argparse

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--action', type=str, default='parse_video')
    parse.add_argument('--video_path', type=str, default='')
    parse.add_argument('--gloss', type=str, default='action')

    args = parse.parse_args()
    action = args.action


    match action:
        # parses a single video, given argument video_path
        case 'parse_video':
            video_parser = VideoParser()
            video_parser.parse_video(args.video_path)
        # create a dataset using all videos in data folder (Note: takes ~ 4 hours to run this on all data)
        case 'create_dataset':
            video_parser = VideoParser()
            video_parser.create_dataset()
        case 'render_all':
            video_renderer = GestureRenderer()
            video_renderer.fetch_all_data()
        case 'render':
            video_renderer = GestureRenderer()
            video_renderer.render_gesture_from_gloss(args.gloss)



if __name__ == '__main__':
    main()
