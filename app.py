from flask import Flask, request, jsonify
import os
import uuid
import shutil
from analyze import process_video

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    video_url = data.get("video_url")
    video_id = data.get("video_id", str(uuid.uuid4()))

    temp_dir = os.path.join("temp", video_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        result_text = process_video(video_url, temp_dir)

        return jsonify({
            "video_id": video_id,
            "status": "success",
            "analysis_result": result_text
        })

    except Exception as e:
        return jsonify({
            "video_id": video_id,
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
