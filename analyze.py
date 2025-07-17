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
    total_image_bytes = 0
    total_estimated_tokens = 0

    for img_path in image_paths:
        file_size = os.path.getsize(img_path)
        total_image_bytes += file_size

        with open(img_path, "rb") as f:
            raw_bytes = f.read()
            img_data = base64.b64encode(raw_bytes).decode("utf-8")

        estimated_tokens = int(len(img_data) / 4)  # Base64大约每4字符=1 token
        total_estimated_tokens += estimated_tokens

        logger.info(f"[IMAGE] {os.path.basename(img_path)}: {file_size / 1024:.2f} KB → ~{estimated_tokens} tokens")

        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
        })

    logger.info(f"[SUMMARY] Total image payload: {total_image_bytes / 1024:.2f} KB")
    logger.info(f"[SUMMARY] Estimated image tokens: ~{total_estimated_tokens}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional ski coach. Your job is to analyze the skier's performance "
                "based on a sequence of video frame images provided by the user. "
                "Only return structured feedback in JSON as instructed. "
                "Do not reject input unless it clearly violates safety policies."
            )
        },
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

        usage = response.usage
        logger.info(f"[TOKENS] Prompt tokens : {usage.prompt_tokens}")
        logger.info(f"[TOKENS] Completion tokens : {usage.completion_tokens}")
        logger.info(f"[TOKENS] Total tokens : {usage.total_tokens}")

        return response.choices[0].message.content

    except Exception as e:
        logger.error("OpenAI 请求失败：", e)
        raise e

from PIL import Image

def compress_image(img_path, max_width=512,quality=70):
    img = Image.open(img_path)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    img.save(img_path, quality=quality)  # 调低质量压缩


def compress_images_to_target_size(image_paths, target_bytes=665600):
    current_width = 512
    quality = 70
    min_width = 256
    round_count = 0

    while True:
        total_bytes = sum(os.path.getsize(p) for p in image_paths)
        logger.info(f"[PROTECT] Compression round {round_count}: total size = {total_bytes / 1024:.2f} KB")

        if total_bytes <= target_bytes or current_width <= min_width:
            logger.info(f"[PROTECT] Compression complete at width={current_width}, size={total_bytes / 1024:.2f} KB")
            break

        current_width = int(current_width * 0.85)  # 缩小比例
        quality = max(quality - 5, 50)  # 降低质量但不低于50

        for p in image_paths:
            compress_image(p, max_width=current_width, quality=quality)

        round_count += 1

def process_video(video_url, temp_dir,category,standard,type_):
    from urllib.parse import urlparse
    video_ext = os.path.splitext(urlparse(video_url).path)[-1] or ".mov"
    video_path = os.path.join(temp_dir, f"input{video_ext}")

    # 步骤 1：下载视频
    download_video(video_url, video_path)

    # 步骤 2：抽帧
    frame_paths = extract_frames(
        video_path=video_path,
        output_dir=temp_dir,
        fps=1.5,
        max_frames=20
    )

    for frame_path in frame_paths:
        compress_image(frame_path)

    compress_images_to_target_size(frame_paths, target_bytes=665600)

    # 步骤 3：调用 OpenAI 分析
    prompt = (
        "You are a professional ski coach. Analyze the skier's performance based on the following sequence of video frames.\n"
        "The analysis context is:\n"
        f"- Category: {category} (ski or snowboard), determines equipment, stance, and technique focus.\n"
        f"- Standard: {standard} (general, aasi, basi, casi), determines the teaching standard and terminology you should apply.\n"
        f"- Type: {type_} (flow or carving), determines the focus of the analysis (overall fluidity vs. edge control and carving technique).\n\n"
        "Please return your analysis in structured JSON format with the following fields:\n\n"
        "1. issue_count: The total number of **distinct issue types**, using technical category **codes** below:\n"
        "   0 - edge transitions\n"
        "   1 - center of gravity\n"
        "   2 - body coordination\n"
        "   3 - pole usage\n"
        "   4 - stance width\n"
        "(Use only the numeric code for type. Do not use vague terms like 'poor posture'.)\n\n"

        "2. issues: A list of objects, each describing one distinct issue. Each issue must include:\n"
        "   - type: One of the codes (0–4) above.\n"
        "   - description: A concise explanation of what was wrong.\n"
        "   - time: The timestamp (in seconds) of the **first frame** where this issue appears.\n"
        "     (If frame 3 shows the issue and frames are 1.5s apart, time = 3.0s)\n"
        "     (If frames 4–6 show the same issue, group them as one and use frame 4 → time = 4.5s)\n"
        "   - suggestion: A clear and specific coaching recommendation to correct the issue.\n\n"

        "Return only:\n"
        "- issue_count\n"
        "- issues (with fields: type, description, time, suggestion)\n"
        "Do NOT return any score or score_reason.\n\n"

        "Return your answer as valid **JSON**, for example:\n"
        "{\n"
        "  \"issue_count\": 2,\n"
        "  \"issues\": [\n"
        "    {\n"
        "      \"type\": 1,\n"
        "      \"description\": \"Skier leans forward in frames 3–5, causing instability.\",\n"
        "      \"time\": 3.0,\n"
        "      \"suggestion\": \"Shift hips slightly backward to align center of mass above the feet.\"\n"
        "    },\n"
        "    {\n"
        "      \"type\": 0,\n"
        "      \"description\": \"Late edge switch in frame 9 increases turn delay.\",\n"
        "      \"time\": 9.0,\n"
        "      \"suggestion\": \"Initiate earlier edge engagement before entering the fall line.\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Return only the raw JSON content. Do not add any explanation, heading, or text outside the JSON."
    )

    return call_gpt4o_with_images(frame_paths, prompt)
