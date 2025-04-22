import flask
import webview
from flask import Response, Flask, request, jsonify
from flask_cors import CORS
import threading
import re
import base64
import numpy as np
import cv2
from renderer import GestureRenderer
import sys
import os
from src.model.english_to_gloss import *

nltk.download('averaged_perceptron_tagger_eng')

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = "./"

    return os.path.join(base_path, relative_path)
app = Flask(__name__)
CORS(app)

mapping_path = resource_path("gloss_to_gesture_mapping_condensed.csv")
landmark_path = resource_path("landmark_data")

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5TokenizerFast
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import torch

saved_model_path = "./checkpoint-29604"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {saved_model_path}...")
loaded_tokenizer = T5TokenizerFast.from_pretrained(saved_model_path)
model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
model.to(device)




try:
    renderer = GestureRenderer(dict_path=mapping_path,
                               landmark_dir=landmark_path)
    print('renderer created')
    print(f'Renderer gloss mapping: {renderer.gloss_gesture_mapping.keys()}')
except Exception as e:
    print(f"ERROR: Failed to initialize GestureRenderer: {e}")
    renderer = None


def start_flask():
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

@app.route('/translate', methods=['POST'])
def translate_sentence():
    # fetch the sentence
    data = request.get_json()
    sentence = data['sentence']
    # process the sentence
    processed_sentence = "translate English to ASL gloss:"+ sentence.lower()
    processed_sentence = re.sub(r'[^\w\s]', '', processed_sentence)
    # this array holds glosses to be rendered, if a gloss is not found, its letters are added to the array individually
    glosses_to_render = []
    known_glosses = set(renderer.gloss_gesture_mapping.keys())
    input_ids = loaded_tokenizer(processed_sentence, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            repetition_penalty=1.9
        )

    translated_sentence = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    print(translated_sentence)
    translated_sentence = re.sub(r"x-", "", translated_sentence)
    translated_sentence = re.sub(r"desc-gloss", "", translated_sentence)
    translated_sentence = re.sub(r"desc-latter", "", translated_sentence)
    translated_sentence = re.sub(r"desc-", "", translated_sentence)
    translated_sentence = re.sub(r"gloss", "", translated_sentence)


    print(translated_sentence)
    print('-'*30)
    print(f"Original sentence: {processed_sentence}, translated sentence: {translated_sentence}")
    # remove x-
    hide_word = False

    for word in translated_sentence.split():
        if hide_word:
            hide_word = False
            continue
        # try to find the gloss from the gloss_gesture_mapping
        if word in known_glosses:
            print(f"Found known gloss: {word}")
            glosses_to_render.append(word)
        else:
            # get index of word from translated_sentence
            try:
                word_index = translated_sentence.find(word)
                comb_word = word + " " + translated_sentence.split()[word_index+1]
                if comb_word in known_glosses:
                    print(f"Found known combined gloss: {comb_word}")
                    glosses_to_render.append(comb_word)
                    hide_word = True
                else:
                    print(f"Unknown word '{word}', spelling word...")
                    # add individual letters to the array to be rendered
                    for letter in word:
                        if letter in known_glosses:
                            glosses_to_render.append(letter)
            except Exception as e:
                print(f"ERROR: Failed to translate: {e}")


    print(glosses_to_render)
    # return the array of glosses
    return jsonify({"glosses_to_render": glosses_to_render})


@app.route('/get_gesture_frames/<gloss>')
def get_gesture_frames(gloss):
    print(f"Rendering: {gloss}")
    frames_base64 = []
    try:
        # try to get the frame from video renderer
        frame_generator = renderer.render_gesture_from_gloss(gloss)

        for frame_bytes in frame_generator:
            # convert to base64 for rendering
            parts = frame_bytes.split(b'\r\n\r\n')
            if len(parts) > 1:
                jpeg_bytes = parts[1].rsplit(b'\r\n', 1)[0]
                frame_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
                frames_base64.append(frame_b64)

        if not frames_base64:
             print(f"Error trying to render {gloss}")
             return jsonify({"error": f"No frames found for '{gloss}'"}), 404
        else:
             print(f"Successfully generated frames for: {gloss}")
             return jsonify({"frames": frames_base64})

    except Exception:
        return jsonify({"error": f"Server error"}), 500

@app.route('/gesture/<gloss>')
def stream_gesture(gloss):
    generator = renderer.render_gesture_from_gloss(gloss)
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    if renderer is None:
        print("Cannot start application because GestureRenderer failed to initialize.")
    else:
        # Start Flask in a separate thread
        print("Starting Flask thread...")
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()

        # Run WebView in the main thread
        print("Starting GUI window...")
        webview.create_window("Sign Language Translator GUI",
                              "./web-app/index.html",
                              width=1024, height=768,
                              resizable=True,
                              maximized=True)
        webview.start(debug=True)
        print("GUI closed, shutting down.")
