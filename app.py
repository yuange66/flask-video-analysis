from flask import Flask, request, jsonify
import os
import uuid
import shutil
import logging
from analyze import process_video

app = Flask(__name__)

# 设置日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info("======== [START] Analyze Video Request ========")

    data = request.get_json()
    logger.info(f"[REQUEST] Received data: {data}")

    video_url = data.get("video_url")
    video_id = data.get("video_id", str(uuid.uuid4()))
    logger.info(f"[INFO] video_url = {video_url}")
    logger.info(f"[INFO] video_id = {video_id}")

    temp_dir = os.path.join("temp", video_id)
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"[INFO] Temporary working directory created: {temp_dir}")

    try:
        # 调用核心分析逻辑
        result_text = process_video(video_url, temp_dir)

        logger.info(f"[RESULT] GPT Analysis result (raw): {result_text}")

        return jsonify({
            "video_id": video_id,
            "status": "success",
            "analysis_result": result_text
        })

    except Exception as e:
        logger.error(f"[ERROR] Exception during analysis: {str(e)}", exc_info=True)
        return jsonify({
            "video_id": video_id,
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"[CLEANUP] Temporary directory deleted: {temp_dir}")
        logger.info("======== [END] Analyze Video Request ========\n")


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
