pyinstaller --onefile ^
    --add-data "web-app;web-app" ^
    --add-data "landmark_data;landmark_data" ^
    --add-data "gloss_to_gesture_mapping_condensed.csv;." ^
    --add-data "gloss_to_gesture_mapping.csv;." ^
    --add-data "checkpoint-29604;checkpoint-29604" ^
    app.py
