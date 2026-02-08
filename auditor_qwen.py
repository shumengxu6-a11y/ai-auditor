import os
# Force clear proxies to avoid DashScope connection errors (Zombie Proxy issue)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import cv2
import whisper
import dashscope
import json
import time
import base64
from http import HTTPStatus

# -----------------------------------------------------------------------------
# Asset A: 私有法规库
# -----------------------------------------------------------------------------
TRAINPAL_RULES = [
  { "id": "R001", "category": "竞品与品牌", "severity": "CRITICAL", "description": "禁止出现竞品Logo（如LNER, Avanti）或画面。", "triggers": ["LNER", "Avanti", "CrossCountry"] },
  { "id": "R002", "category": "价格合规", "severity": "HIGH", "description": "禁止使用'最便宜'、'最低价'等绝对化表述，除非有证据。", "triggers": ["Cheapest", "Lowest Price", "Best", "无敌"] },
  { "id": "R003", "category": "政治敏感", "severity": "CRITICAL", "description": "禁止将'香港'与'中国'并列，禁止表述'去往中国'，必须符合一个中国原则。", "triggers": ["香港去往中国", "Hong Kong and China"] },
  { "id": "R004", "category": "不文明画面", "severity": "HIGH", "description": "禁止出现踢箱子、抢耳机、铁路脏乱差画面。", "triggers": ["踢箱子", "抢耳机", "脏乱"] }
]

# -----------------------------------------------------------------------------
# Asset B: Qwen System Prompt
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
你是 TrainPal 出海营销团队的 AI 内容合规审核助手。你必须以**零容忍 (Zero Tolerance)** 的态度执行以下 12 条绝对红线审核标准。

请结合 [Visual Analysis]（视觉画面）和 [Audio Transcript]（音频文本），逐帧逐句严格排查。

---
🚨 **12 条绝对红线 (Absolute Red Lines)**
---

**1️⃣ 竞品与品牌关系**
❌ 视频画面中出现任何火车运营商 Logo（LNER, Avanti, CrossCountry, GWR 等）→ **FAIL**
❌ 出现竞品品牌 Logo 或画面素材 → **FAIL**
❌ 踩竞品品牌（画面打叉其他品牌、文案说其他品牌不好）→ **FAIL**
✅ 只能展示 TrainPal 自己的产品截图，且截图中不能有运营商 Logo

**2️⃣ 价格与优惠合规**
❌ 使用绝对化表述："最便宜"、"最低价格"、"最佳"、"领先"、"最棒"、"无敌" → **FAIL**
❌ 虚假价格 P 图、虚构原价 → **FAIL**
❌ 未加限定条件的优惠承诺（如"伦敦到曼城10镑直达"）→ **FAIL**
✅ 必须加限定条件："10镑起（提前21天预订/低峰时段）"

**3️⃣ 政治与敏感内容**
❌ 将"香港"与"中国"并列或对立（如"香港和中国"、"Hong Kong and China"）→ **FAIL**
❌ 种族/性别/阶级歧视内容 → **FAIL**
❌ 名人肖像侵权（名人 meme、肖像）→ **FAIL**
❌ 涉及黄赌毒、酒精药物、宗教、政治、血腥暴力 → **FAIL**

**4️⃣ 内容真实性**
❌ 冷知识不真实、无法验证 → **FAIL**
❌ 虚假评论/回复用户 → **FAIL**
❌ 误导用户权利（如"改签到17天后再退票就能免费"未说明限制条件）→ **FAIL**

**5️⃣ 广告身份披露**
❌ 昵称/简介提及 TrainPal 品牌 → **FAIL**
❌ HashTag 带 Trip、Trainpal 等话题 → **FAIL**
✅ 账号必须人设化（如大学生、省钱达人、退休人员）

**6️⃣ 肖像权与隐私 [最高优先级]**
❌ **未打码路人正脸/侧脸/半脸** → **FAIL (立即判定)**
   - 只要能看到眼睛、鼻子、嘴巴等面部特征 → **FAIL**
   - 即使人脸很小、在背景中、只出现一瞬间 → **也必须标记违规**
❌ 使用真人网络素材头像 → **FAIL**
❌ 5000粉丝以上 KOL/网红怼脸视频 → **FAIL**
❌ 名人、明星出镜画面 → **FAIL**
❌ 10万以上粉丝 KOL 正脸出镜 → **FAIL**
❌ 儿童肖像（各地区对儿童肖像保护特别严格）→ **FAIL**

**7️⃣ 违法与暴力内容**
❌ 违法犯罪行为（吸毒、赌博、酗酒、偷窃、诈骗）→ **FAIL**
❌ 暴力血腥（凶杀、打架斗殴、自残、车祸现场、手术特写）→ **FAIL**
❌ 恐怖主义（涉及恐怖组织、极端主义、邪教）→ **FAIL**
❌ 饮酒/抽烟画面营造"鼓励、煽动"氛围 → **FAIL**

**8️⃣ 色情低俗**
❌ 明确色情（裸露生殖器官、性行为或模仿性行为）→ **FAIL**
❌ 焦点性暗示（镜头长时间聚焦胸部、臀部、大腿根部）→ **FAIL**
❌ 性挑逗动作（模拟性爱动作、刻意抖胸、撩裙子）→ **FAIL**
❌ 软色情内容（"福利姬"、性暗示 ASMR、尺度极大的舞蹈）→ **FAIL**

**9️⃣ 未成年人保护**
❌ 任何涉及未成年人的情色、软色情内容 → **FAIL**
❌ 恶搞、羞辱未成年人的画面（如校园暴力片段）→ **FAIL**
❌ 泄露未成年人隐私（学校、姓名、家庭住址）→ **FAIL**

**🔟 禁止抹黑铁路服务质量或从业人员**
❌ 暗示铁路乘务员、客服"有意隐瞒"、"态度差"、"不作为" → **FAIL**
❌ 使用"内部价"、"漏洞"、"走后门" → **FAIL**
❌ 暗示官方合作 → **FAIL**

**1️⃣1️⃣ 供应商红线**
❌ "千万不要去火车站购票" → **FAIL**
❌ 线下买票话术 → **FAIL**
❌ 事故/故障画面（有损英国铁路形象的视频素材）→ **FAIL**
❌ 火车供应商负面引导内容 → **FAIL**

**1️⃣2️⃣ 风险画面素材**
❌ 男人踢箱子、抢耳机、女人掉坑里、火车着火、乘客跟随进站等不文明行为 → **FAIL**

---
**判定逻辑 (Strict Logic):**
- 只要发现上述任何一项违规（哪怕只有一帧或一句话），结果必须是 **FAIL**。
- **人脸检测是最高优先级**，请逐帧仔细检查每一张画面。
- 只有当视频完全干净、没有任何风险点时，才能给 **PASS**。
- 风险评分 (Risk Score)：发现违规直接打 **100**，无违规打 **0**。不要给中间分。

输出必须为纯 JSON 格式（不要包含 ```json ... ```）：
{
  "audit_result": "PASS" | "FAIL",
  "risk_score": 0 或 100,
  "violations": [{"timestamp": "xxs", "reason": "发现可识别人脸（眼睛+鼻子可见）/发现LNER Logo/使用绝对化表述'最便宜'/男人踢箱子画面...", "category": "肖像隐私/竞品与品牌/价格合规/风险画面素材..."}]
}
"""

class VideoAuditor:
    def __init__(self, api_key):
        self.api_key = api_key
        dashscope.api_key = api_key
        self.asr_model = None  # To hold the local Whisper model instance

    def _load_whisper(self):
        if self.asr_model is None:
            # Using 'base' model - will auto-download if network permits
            print("⏳ Loading Whisper model (base)...")
            self.asr_model = whisper.load_model("base")

    def extract_audio(self, video_path):
        """
        Extracts audio/transcript using local Whisper (OpenAI).
        Running locally since ffmpeg is now installed!
        """
        try:
            self._load_whisper()
            result = self.asr_model.transcribe(video_path)
            # result['text'] contains the full transcript
            return result['text']
        except Exception as e:
            return f"[Whisper Error] {str(e)}"




    def extract_keyframes(self, video_path, interval_sec=None):
        """
        [自适应帧采样] Adaptive Frame Sampling via Scene Change Detection.
        - Detects significant visual changes (scene cuts).
        - Guaranteed max interval: 4s (to catch static violations).
        - Minimum interval: 1s (to avoid redundancy).
        """
        frames_paths = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25
        
        # Adaptive Parameters (Tuned for Face Detection)
        last_saved_time = -100
        min_interval = 0.8  # Min 0.8s between frames (faster capture)
        max_interval = 3.0  # Max 3s without frame (denser sampling)
        threshold = 18.0    # Lower threshold = more sensitive to subtle changes like faces
        
        prev_gray_frame = None
        
        # Ensure temp dir exists
        temp_dir = "temp_frames_qwen"
        os.makedirs(temp_dir, exist_ok=True)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = frame_idx / fps
            
            # Convert to gray for fast diffing
            small_frame = cv2.resize(frame, (320, 180)) # Resize for speed
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            should_save = False
            time_since_last = current_time - last_saved_time
            
            # 1. First frame always save
            if last_saved_time < 0:
                should_save = True
                
            # 2. Max interval forced save
            elif time_since_last >= max_interval:
                should_save = True
                
            # 3. Scene Change Detection (only if min_interval passed)
            elif time_since_last >= min_interval:
                if prev_gray_frame is not None:
                    # Calculate difference score
                    score = cv2.mean(cv2.absdiff(gray, prev_gray_frame))[0]
                    if score > threshold:
                        should_save = True
            
            if should_save:
                # Save Full Resolution Frame (or slightly resized for API limit)
                # Resize to max 1024px width to save bandwidth/tokens
                h, w = frame.shape[:2]
                if w > 1024:
                    scale = 1024 / w
                    frame = cv2.resize(frame, (1024, int(h * scale)))
                
                # Burn Timestamp into Image (Visual Watermark for AI)
                # Text: "T: 4.2s"
                ts_text = f"T: {current_time:.1f}s"
                cv2.putText(frame, ts_text, (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)

                frame_name = f"{temp_dir}/frame_{current_time:.2f}s.jpg"
                cv2.imwrite(frame_name, frame)
                frames_paths.append((round(current_time, 2), frame_name))
                
                last_saved_time = current_time
                prev_gray_frame = gray  # Update reference for next diff
            
            frame_idx += 1
            
        cap.release()
        print(f"📸 Adaptive Sampling: Extracted {len(frames_paths)} frames from video.")
        return frames_paths

    def audit(self, frames_data, audio_text, model_config=None):
        """
        Multimodal Audit: Supports Qwen (DashScope) and Generic OpenAI-Compatible (DeepSeek, GPT-4o, etc.)
        """
        # Default to Qwen if no config provided
        if model_config is None:
            model_config = {"type": "qwen", "model_name": "qwen-vl-max"}

        # Preparation: Instruction with Transcript
        if audio_text.startswith("[") and ("Error" in audio_text or "Exception" in audio_text):
            audio_instruction = f"\n【Audio Transcript】:\n(音频提取失败: {audio_text})\n\n⚠️ **特别指令**：仅基于视觉画面审核。\n"
        else:
            audio_instruction = f"\n【Audio Transcript】:\n\"{audio_text}\"\n\n请综合分析画面与音频。\n"

        prompt_final = f"{SYSTEM_PROMPT}\n{audio_instruction}\n必须返回纯 JSON。"

        # --- BRANCH A: QWEN (DASHSCOPE SDK) ---
        if model_config['type'] == 'qwen':
            content = []
            for _, path in frames_data:
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    content.append({"image": f"data:image/jpeg;base64,{encoded}"})
            content.append({"text": prompt_final})
            
            try:
                response = dashscope.MultiModalConversation.call(
                    model='qwen-vl-max',
                    messages=[{"role": "user", "content": content}],
                    result_format='message'
                )
                if response.status_code == HTTPStatus.OK:
                    raw = response.output.choices[0].message.content
                    if isinstance(raw, list): raw = "".join([i['text'] for i in raw if 'text' in i])
                    return raw.replace("```json", "").replace("```", "").strip()
                return json.dumps({"audit_result": "ERROR", "error_msg": response.message})
            except Exception as e:
                return json.dumps({"audit_result": "ERROR", "error_msg": str(e)})

        # --- BRANCH B: OPENAI-COMPATIBLE (REQUESTS) ---
        else:
            import requests
            api_key = model_config.get('api_key')
            base_url = model_config.get('base_url', 'https://api.openai.com/v1').rstrip('/')
            model_name = model_config.get('model_name', 'gpt-4o')
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # OpenAI Multimodal Content Format
            payload_content = [{"type": "text", "text": prompt_final}]
            
            for _, path in frames_data:
                # OPTIMIZATION: Resize & Compress to avoid 413 Entity Too Large
                # DeepSeek/OpenAI Gateway usually limits body size (e.g., 5MB-10MB).
                # Raw images will easily exceed this.
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        # 1. Resize max dimension to 768px (Sufficient for audit)
                        h, w = img.shape[:2]
                        max_dim = max(h, w)
                        if max_dim > 768:
                            scale = 768 / max_dim
                            img = cv2.resize(img, (int(w*scale), int(h*scale)))
                        
                        # 2. Compress to JPEG with Quality 60
                        # This reduces size from ~500KB to ~30-50KB per frame
                        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        b64_img = base64.b64encode(buffer).decode('utf-8')
                        
                        payload_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })
                except Exception as e:
                    print(f"Skipping frame {path} due to error: {e}")
            
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": payload_content}],
                "max_tokens": 1024
            }
            
            try:
                resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
                
                if resp.status_code == 200:
                    raw = resp.json()['choices'][0]['message']['content']
                    return raw.replace("```json", "").replace("```", "").strip()
                
                elif resp.status_code == 400 and ("image_url" in resp.text or "variant" in resp.text):
                     # Specific handling for models that don't support Vision (e.g., DeepSeek-V3 text-only)
                     raise Exception(f"Model does not support Vision inputs (400 Bad Request).")
                     
                else:
                    return json.dumps({"audit_result": "ERROR", "error_msg": f"API Error {resp.status_code}: {resp.text}"})
                    
            except Exception as e:
                # Engineering Best Practice: Failover Mechanism
                # If specialized model fails (network issue or capability issue), fallback to stable internal model
                print(f"⚠️ [Failover Triggered] Issue with {base_url}: {e}")
                print("🔄 Automatically switching to Reserve Model (Qwen-VL-Max)...")
                
                # Recursive call with Qwen config
                fallback_config = {"type": "qwen", "model_name": "qwen-vl-max (Fallback)"}
                return self.audit(frames_data, audio_text, model_config=fallback_config)


    @staticmethod
    def estimate_cost(token_usage):
        pass
