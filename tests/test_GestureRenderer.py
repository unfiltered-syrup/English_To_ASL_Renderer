from src.renderer.GestureRenderer import *

def test_gesture_renderer_init_success():
    # create a dummy CSV file for testing
    dummy_data = {'gloss': ['hello', 'world'], 'landmark_file': ['hello.npz', 'world.npz']}
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = "temp_gloss_to_gesture_mapping.csv"
    dummy_df.to_csv(dummy_csv_path, index=False)

    renderer = GestureRenderer(dict_path=dummy_csv_path)
    assert isinstance(renderer.gloss_gesture_mapping, dict)
    assert 'hello' in renderer.gloss_gesture_mapping
    assert renderer.gloss_gesture_mapping['hello'] == {'landmark_file': 'hello.npz'}
    assert renderer.landmark_dir == "./landmark_data/"
    assert renderer.canvas_width == 640
    assert renderer.canvas_height == 520
    assert isinstance(renderer.mp_drawing, mp.solutions.drawing_utils.DrawingUtils)
    assert isinstance(renderer.mp_hands, mp.solutions.hands.Hands)

    # remove the dummy file
    os.remove(dummy_csv_path)
