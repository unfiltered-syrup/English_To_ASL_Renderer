import flask
import webview
from flask import Response, Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)

CORS(app)

def start_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)


if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Run WebView in the main thread
    webview.create_window("GUI", "./web-app/index.html", width=1024, height=768, resizable=True,
                          fullscreen=False, maximized=True)
    webview.start()