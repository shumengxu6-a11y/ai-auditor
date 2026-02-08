import os

import base64
import time
from moviepy import VideoFileClip
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import io

# ==========================================
# 核心业务规则 (Business Rules)
# ==========================================
# ==========================================
# 核心业务规则 (Business Rules)
# ==========================================
SYSTEM_PROMPT = """
你是一位专业的"TrainPal 短视频合规审核专家"。你的任务是根据以下《TrainPal 合规红线》严格审查视频的画面、字幕及语音内容。

请务必从以下 **3 个独立维度** 进行交叉审核，并指出具体的违规时间点：

### 1. 👁️ 违规画面 (Visual)
- **竞品排他**: 严禁出现 "National Rail", "LNER", "Trainline" 等非 TrainPal 标志。
- **不文明行为**: 严禁 "脚踩座椅", "抢占耳机/手机", "醉酒/吸烟"。
- **安全隐患**: 严禁出现火车着火、事故、由于延误导致的混乱场面。

### 2. 📝 违规字幕 (Subtitle / OCR)
- **价格合规**: 禁止绝对化描述（如 "Cheapest", "No.1"）；必须包含条件前缀（"From £10", "Up to 50% off"）。
- **虚假承诺**: 如 "Delay Repay" 必须带 "Subject to T&Cs"。
- **政治/敏感词**: 严禁涉及香港/台湾政治问题及种族歧视内容。

### 3. 🎤 违规配音 (Audio / Dubbing)
- **语音内容**: 必须审核配音内容是否包含辱骂、诱导消费或政治敏感词。
- **音画一致**: 配音承诺必须与字幕条款一致（例如配音说"全额退款"但字幕写"部分退款"即为违规）。

**输出格式要求 (JSON)**：
请严格输出 JSON 格式，不要包含 Markdown 代码块标记：
{
  "is_compliant": true/false,
  "risk_score": 0-100,
  "issues": [
    {
      "timestamp": "MM:SS (例如 00:04)",
      "dimension": "画面 / 字幕 / 配音",
      "category": "违规类别 (如: 竞品排他)",
      "description": "详细描述违规内容 (例如: 画面左上角出现 LNER 图标)",
      "suggestion": "具体的修改建议 (例如: 使用高斯模糊遮盖 LNER 图标)"
    }
  ]
}
如果视频完全合规，issues 数组为空，risk_score 为 0。
"""

# ==========================================
# 音频与视频处理 (Audio & Video Processing)
# ==========================================

def get_video_duration(video_path):
    clip = VideoFileClip(video_path)
    dur = clip.duration
    clip.close()
    return dur

def extract_audio(video_path, output_path="temp_audio.mp3"):
    """
    从视频中提取音频
    """
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path, logger=None)
        clip.close()
        return output_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None

def extract_frames(video_path, fps=1, output_folder="temp_frames"):
    """
    从视频中每秒提取 1 帧。
    返回提取的图片路径列表。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 清理旧文件
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame_paths = []
    
    print(f"Video duration: {duration}s. Extracting 1 frame per second...")
    
    for t in range(0, int(duration) + 1):
        frame_path = os.path.join(output_folder, f"frame_{t}.jpg")
        clip.save_frame(frame_path, t=t)
        frame_paths.append(frame_path)
    
    clip.close()
    return frame_paths, duration

# ==========================================
# AI 分析引擎 (Analysis Engine)
# ==========================================

def gemini_upload(video_path, api_key):
    """
    Step 1: Upload video to Gemini
    """
    genai.configure(api_key=api_key)
    print(f"Uploading {video_path} to Gemini...")
    return genai.upload_file(path=video_path)

def gemini_get_file(file_name):
    """
    Helper to get file status without blocking.
    """
    return genai.get_file(file_name)

def gemini_wait_for_processing(video_file, sleep_interval=2):
    """
    Step 2: Wait for video processing (blocks until ACTIVE)
    """
    while video_file.state.name == "PROCESSING":
        time.sleep(sleep_interval)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    return video_file

def gemini_generate_report(video_file, model_name="gemini-2.5-flash"):
    """
    Step 3: Analyze content
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [SYSTEM_PROMPT, video_file],
        request_options={"timeout": 600}
    )
    return response.text

def analyze_video_gemini_native(video_path, api_key, model_name="gemini-2.5-flash"):
    """
    (Legacy Wrapper for backward compatibility if needed, but app.py should use new steps)
    """
    f = gemini_upload(video_path, api_key)
    f = gemini_wait_for_processing(f)
    return gemini_generate_report(f, model_name)

def transcribe_audio(audio_path, api_key, base_url):
    """
    使用 OpenAI 兼容接口 (Whisper) 进行语音转文字
    **Upgrade**: 使用 verbose_json 获取细粒度的时间戳 (Segments)，实现精准的音画同步审核。
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    audio_file = open(audio_path, "rb")
    try:
        # Request verbose_json to get segments and timestamps
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        return transcript # Return the full object/dict
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def get_transcript_segment(transcript_obj, current_time):
    """
    根据当前视频时间戳，从 Whisper 结果中提取对应的字幕片段。
    """
    if not transcript_obj:
        return ""
    
    # Check if object or dict
    if hasattr(transcript_obj, 'segments'):
        segments = transcript_obj.segments
    elif isinstance(transcript_obj, dict) and 'segments' in transcript_obj:
        segments = transcript_obj['segments']
    else:
        return getattr(transcript_obj, 'text', str(transcript_obj)) # Fallback to full text

    context_text = []
    # 查找覆盖 current_time 的片段，或者前后 2 秒的片段 (Context Window)
    for seg in segments:
        start = seg.get('start') if isinstance(seg, dict) else seg.start
        end = seg.get('end') if isinstance(seg, dict) else seg.end
        text = seg.get('text') if isinstance(seg, dict) else seg.text
        
        # 如果当前时间落在片段范围内，或者非常接近
        if start <= current_time + 1 and end >= current_time - 1:
            context_text.append(f"[{start:.1f}s - {end:.1f}s]: {text}")
            
    return "\n".join(context_text)

def analyze_frame_gemini(image_path, api_key, model_name="gemini-2.5-flash"):
    # (保留用于 fallback)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    img = Image.open(image_path)
    try:
        response = model.generate_content([SYSTEM_PROMPT, img])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_frame_openai_compatible(image_path, api_key, base_url, model_name="gpt-4o", audio_context=None):
    """
    使用 OpenAI 兼容接口 (如 DeepSeek, Moonshot 等)
    新增 audio_context 以支持语音转写内容的语义审核。
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Build prompt
    user_content_list = [
        {"type": "text", "text": "请审核这张视频截图是否合规。"}
    ]
    
    if audio_context:
        user_content_list.append({
            "type": "text",
            "text": f"\n\n【附加多模态信息 (Audio Transcript)】\n当前视频片段的语音转文字内容如下，请结合这些内容辅助审核配音违规与音画一致性：\n{audio_context}"
        })

    user_content_list.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content_list
                }
            ],
            max_tokens=2000 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def estimate_cost(duration_seconds, model_name="gemini-2.5-flash"):
    """
    估算成本 (Based on 2025 Public Pricing - Estimates only)
    Ref: https://ai.google.dev/pricing
    Ref: https://openai.com/pricing
    """
    # Token Estimation
    # Video: Gemini charges ~263 tokens/second. GPT-4o Vision: ~85-170 tokens/frame (low res) or 1000+ (high res).
    # Audio: Whisper ~ $0.006 / min.
    
    total_cost = 0.0
    details = ""
    
    # 1. Google Gemini Pricing (Flash is extremely cheap)
    if "gemini" in model_name.lower():
        if "flash" in model_name.lower():
            price_per_1m_tokens = 0.10 # $0.10 input / 1M tokens
        else: # Pro
            price_per_1m_tokens = 2.50 # $2.50 input / 1M tokens (approx)
            
        estimated_tokens = duration_seconds * 300 # Video + Audio tokens
        total_cost = (estimated_tokens / 1_000_000) * price_per_1m_tokens
        details = f"Gemini ({model_name}): {estimated_tokens} tokens @ ${price_per_1m_tokens}/1M"

    # 2. OpenAI / DeepSeek Pricing
    else:
        # GPT-4o / DeepSeek (Vision + Whisper)
        frames = duration_seconds # 1 FPS
        
        # Whisper Cost
        whisper_cost = (duration_seconds / 60) * 0.006
        
        # Vision Cost (Assumption: Low-res detail for cost efficiency or High-res for quality)
        # DeepSeek is cheaper, GPT-4o is expensive.
        if "deepseek" in model_name.lower():
            # DeepSeek V3/R1 is approx $0.14-$0.55 / 1M tokens. 
            # Vision capabilities via API vary, assuming similar token mapping.
            vision_price_per_image = 0.0002 
        elif "mini" in model_name.lower():
            vision_price_per_image = 0.0005 # GPT-4o-mini
        else:
            vision_price_per_image = 0.005 # GPT-4o Standard (expensive)
            
        vision_cost = frames * vision_price_per_image
        total_cost = whisper_cost + vision_cost
        details = f"{model_name}: Whisper(${whisper_cost:.4f}) + Vision({frames}f * ${vision_price_per_image})"
        
    return total_cost, details 
