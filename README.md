# English To ASL Renderer

## Overview

An American Sign Language translation and rendering pipeline. 

![FlowChart](Flowchart.png "Flow Chart")

## Features 🌟

* **English Text Input:** Simple interface to enter English sentences or phrases.
* **ASL Gloss Translation:** Leverages a fine-tuned T5 model to translate English into a suitable ASL representation (e.g., gloss).
* **Gesture Rendering:** Visualizes the translated ASL using MediaPipe Hand landmarks for realistic hand shapes and movements.
* **Interactive Learning:** Provides a visual aid for understanding how English concepts map to ASL signs.

## How It Works ⚙️

This project combines natural language processing with computer vision techniques:

1.  **Frontend (WebView):** A clean web interface allows users to input English text.
2.  **Backend (Flask):** The Flask server receives the text input from the frontend.
3.  **Translation (Fine-tuned T5):** The input text is passed to a fine-tuned T5 model specialized for English-to-ASL gloss translation.
4.  **Gesture Logic (Backend):** The backend interprets the generated ASL gloss and calculates the corresponding sequence of hand poses and movements, likely referencing MediaPipe Hand landmark definitions.
5.  **Rendering (WebView + MediaPipe):** The pose sequence data is sent back to the frontend. The WebView interface uses this data, potentially leveraging MediaPipe's visualization utilities or custom rendering logic based on MediaPipe landmarks, to display the animated ASL hand gestures.

## Tech Stack 🔧

* **Machine Translation:** Fine-tuned `T5` model
* **Backend Framework:** `Flask`
* **Frontend Interface:** `WebView`
* **Hand Tracking/Rendering:** `MediaPipe Hands`

## Getting Started 🚀

Follow these steps to get the project running locally.

### Prerequisites

* Python 3.8+ and Pip
* install "requirements.txt"
* 
### Installation

1.  **Clone the repository:**

2.  **Set up Python environment (Recommended):**

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download model from this link: https://www.dropbox.com/scl/fo/35g0b131765pte1b5rowi/AAmkHVrfDmpOQQ_6cp7D0Pk?rlkey=hcliy2jamdlb8kqfuozedya2y&st=wuxkb0al&dl=0**

5.  **Drop the fine-tuned T5 model in the root folder:**

### Running the Application

```bash
    python app.py
```

I've included a .exe file for Windows users if set up is too complicated.

First time running the executable may be slow since it will attempt to install all dependencies.


## Acknowledgements 🙏

* Google's [MediaPipe](https://developers.google.com/mediapipe) team for the hand tracking toolkit.
* The Hugging Face team for the `transformers` library and T5 model access.