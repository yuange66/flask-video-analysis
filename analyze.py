import os
import subprocess
import requests
import base64
import openai
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# openai.api_key = "OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

def download_video(video_url, save_path):
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def extract_frames(video_path, output_dir, fps=1, max_frames=10):
    frame_pattern = os.path.join(output_dir, "frame_%03d.jpg")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-vframes", str(max_frames),
        frame_pattern
    ]
    subprocess.run(cmd, check=True)

    return [
        os.path.join(output_dir, f)
        for f in sorted(os.listdir(output_dir))
        if f.endswith(".jpg")
    ]


def call_gpt4o_with_images(image_paths, prompt):
    images = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
            })

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + images
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )

        #  打印 token 使用情况
        usage = response.usage
        logger.info(f"  Token usage:")
        logger.info(f"  Prompt tokens : {usage.prompt_tokens}")
        logger.info(f"  Completion tokens : {usage.completion_tokens}")
        logger.info(f"  Total tokens : {usage.total_tokens}")

        return response.choices[0].message.content

    except Exception as e:
        logger.error("OpenAI 请求失败：", e)
        raise e


def process_video(video_url, temp_dir):
    video_path = os.path.join(temp_dir, "input.mp4")

    # 步骤 1：下载视频
    download_video(video_url, video_path)

    # 步骤 2：抽帧
    frame_paths = extract_frames(
        video_path=video_path,
        output_dir=temp_dir,
        fps=1.5,
        max_frames=20
    )

    # 步骤 3：调用 OpenAI 分析
    prompt = (
        "You are a professional ski coach. Analyze the skier's performance based on the following sequence of video frames.\n"
        "Please return your analysis in structured form including the following:\n\n"
        "1. issue_count: The total number of problems or risks observed (e.g., poor posture, unstable balance, risk of falling).\n"
        "2. issues: A list of brief descriptions of each identified issue.\n"
        "3. score: Give a score from 0 to 100 based on the skier’s technique, consistency, and safety.\n"
        "4. score_reason: A short explanation of why this score was given.\n\n"
        "Return your answer in **English** and **JSON format** like:\n"
        "{\n"
        "  \"issue_count\": 2,\n"
        "  \"issues\": [\n"
        "    \"Skier leans too far forward in frame 3, causing instability.\",\n"
        "    \"Leg position too wide in frame 9, risking edge control.\"\n"
        "  ],\n"
        "  \"score\": 78,\n"
        "  \"score_reason\": \"Overall good movement, but two technical flaws impact safety and form.\"\n"
        "}"
    )
    return call_gpt4o_with_images(frame_paths, prompt)
