from preprocessing import VideoParser
import argparse

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--action', type=str, default='parse_video')
    parse.add_argument('--video_path', type=str, default='')

    args = parse.parse_args()
    action = args.action


    match action:
        # parses a single video, given argument video_path
        case 'parse_video':
            video_parser = VideoParser()
            video_parser.parse_video(args.video_path)
        # create a dataset using all videos in data folder (in progress)
        case 'create_dataset':
            video_parser = VideoParser()
            video_parser.create_dataset()


if __name__ == '__main__':
    main()
