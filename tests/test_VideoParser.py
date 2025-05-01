from src.preprocessing.VideoParser import *

import mediapipe as mp


def test_init_default():
    parser = VideoParser()
    assert isinstance(parser.hands, mp.solutions.hands.Hands)
    assert not parser.hands.static_image_mode
    assert parser.hands.max_num_hands == 2
    assert parser.hands.min_detection_confidence == 0.3
    assert parser.target_height == 520
    assert not parser.debug_mode


def test_init_custom():
    parser = VideoParser(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, debug_mode=True)
    assert parser.hands.static_image_mode
    assert parser.hands.max_num_hands == 1
    assert parser.hands.min_detection_confidence == 0.5
    assert parser.debug_mode
