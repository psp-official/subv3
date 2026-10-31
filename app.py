#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render Paid Plan - Subtitle Generator Web Service (app.py)
==========================================================
- Mode 1 (original): API Key မလိုဘဲ Faster-Whisper (int8 CPU optimized) ဖြင့် မူရင်းဘာသာ .srt ထုတ်ပေးသည်။
- Mode 2 (myanmar) : Gemini API ဖြင့် မြန်မာစာတန်းထိုး (.srt) တိုက်ရိုက်ထုတ်ပေးသည်။
- Mode 3 (both)    : Faster-Whisper ဖြင့် မူရင်း .srt အရင်ထုတ်ပြီး Gemini API ဖြင့် မြန်မာ .srt သို့ ဘာသာပြန်ပေးသည်။
- Render Cloudflare HTTP 100s Timeout မဖြစ်စေရန် Background Job Queue + Live Progress Polling စနစ် ပါဝင်သည်။
"""

import os
import re
import uuid
import time
import glob
import shutil
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)

# Render Paid Plan တွင် ဗီဒီယိုဖိုင်အကြီးများ တင်နိုင်ရန် (Max 500MB)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

JOBS_DIR = Path(tempfile.gettempdir()) / "render_sub_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (Use 1 Gunicorn worker with multiple threads on Render)
JOBS: Dict[str, Dict[str, Any]] = {}
WHISPER_CACHE: Dict[str, Any] = {}
WHISPER_LOCK = threading.Lock()

# =====================================================================
# 1. SYSTEM PROMPTS FOR GEMINI MYANMAR TRANSLATION
# =====================================================================

AUDIO_TO_MM_SRT_SYSTEM_PROMPT = """# Role & Objective
You are an elite Multimodal Subtitle Generator and Audiovisual Translator.
Your mission is to listen to the uploaded foreign movie/series/donghua audio clip, accurately transcribe the spoken dialogue with precise timestamps, and translate it directly into **short, punchy, natural, and culturally accurate Burmese (Myanmar Unicode)** in standard `.srt` format.

# Core Rules (CRITICAL)
1. Output ONLY valid `.srt` subtitle blocks. Do NOT include markdown code blocks (no ```srt).
2. Sync start and end timestamps (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) strictly with actual spoken dialogue.
3. Use natural spoken Burmese suited to the character's tone and relationship.
4. Always use standard Myanmar Unicode (NOT Zawgyi).
"""

SRT_TRANSLATE_SYSTEM_PROMPT = """# Role & Objective
You are a professional Movie Subtitle Translator specializing in translating foreign `.srt` subtitles into **natural, colloquial Burmese (Myanmar Unicode)**.

# Core Rules (CRITICAL)
1. Keep the exact subtitle index numbers and timestamps (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) unchanged.
2. Translate ONLY the dialogue text into natural spoken Burmese.
3. Return ONLY the valid `.srt` blocks without markdown fences or extra commentary.
"""

TIMESTAMP_REGEX = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def time_to_ms(hours: int, minutes: int, seconds: int, ms: int) -> int:
    return ((hours * 3600) + (minutes * 60) + seconds) * 1000 + ms


def ms_to_srt_time(total_ms: int) -> str:
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def seconds_to_srt_time(seconds_float: float) -> str:
    return ms_to_srt_time(int(round(seconds_float * 1000)))


def clean_llm_srt_output(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(?:srt|plaintext|text)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_and_offset_srt(srt_text: str, offset_ms: int, start_index: int = 1) -> Tuple[List[str], int]:
    cleaned = clean_llm_srt_output(srt_text)
    if not cleaned:
        return [], start_index

    blocks = re.split(r"\n\s*\n", cleaned)
    formatted_blocks = []
    current_idx = start_index

    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        ts_line_idx = -1
        match = None
        for i, line in enumerate(lines):
            m = TIMESTAMP_REGEX.search(line)
            if m:
                ts_line_idx = i
                match = m
                break

        if ts_line_idx == -1 or not match:
            continue

        dialogue_lines = lines[ts_line_idx + 1 :]
        if not dialogue_lines:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
        start_time_ms = time_to_ms(h1, m1, s1, ms1) + offset_ms
        end_time_ms = time_to_ms(h2, m2, s2, ms2) + offset_ms

        new_ts_line = f"{ms_to_srt_time(start_time_ms)} --> {ms_to_srt_time(end_time_ms)}"
        dialogue_text = "\n".join(dialogue_lines)

        formatted_blocks.append(f"{current_idx}\n{new_ts_line}\n{dialogue_text}")
        current_idx += 1

    return formatted_blocks, current_idx


# =====================================================================
# 2. FFMPEG & FASTER-WHISPER HELPERS (RENDER PAID PLAN OPTIMIZED)
# =====================================================================

def extract_lightweight_audio(input_media: str, output_mp3: str):
    """Extract 16kHz Mono MP3 via FFmpeg to minimize RAM usage on Render."""
    cmd = [
        "ffmpeg", "-y", "-i", input_media,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        output_mp3
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")


def get_whisper_model(model_size: str = "small"):
    """Singleton pattern to keep Faster-Whisper loaded in RAM across requests."""
    with WHISPER_LOCK:
        if model_size not in WHISPER_CACHE:
            from faster_whisper import WhisperModel
            cpu_threads = int(os.environ.get("WHISPER_THREADS", "4"))
            WHISPER_CACHE[model_size] = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
                download_root=os.path.join(tempfile.gettempdir(), "whisper_models")
            )
        return WHISPER_CACHE[model_size]


def run_whisper_original_srt(job_id: str, audio_path: str, output_srt: str, model_size: str, lang_code: str):
    JOBS[job_id]["message"] = f"Loading Faster-Whisper ({model_size}, int8)..."
    model = get_whisper_model(model_size)

    JOBS[job_id]["message"] = "Transcribing original language with accurate timestamps (No API Key)..."
    language_arg = lang_code.strip() if lang_code and lang_code.strip() else None

    segments, info = model.transcribe(
        audio_path,
        language=language_arg,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True  # <--- အရေးကြီးဆုံး ပြင်ဆင်ချက် (စကားလုံးအလိုက် အချိန်ယူရန်)
    )

    total_dur = max(info.duration, 1.0)
    JOBS[job_id]["detected_lang"] = info.language
    idx = 1

    with open(output_srt, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            
            start_time = seg.start
            end_time = seg.end

            # စကားလုံးတွေရဲ့ အတိအကျ အဆုံးသတ်ချိန်ကို ယူမယ်
            if hasattr(seg, 'words') and seg.words:
                end_time = seg.words[-1].end
                
                # စာတန်းဖတ်ရလွယ်ကူအောင် အနည်းဆုံး ၀.၅ စက္ကန့် ထားပေးမယ်
                if end_time - start_time < 0.5:
                    end_time = start_time + 0.5
                
                # စကားပြောပြီးတာနဲ့ ချက်ချင်းပျောက်သွားအောင် 0.15 စက္ကန့် (150ms) လျှော့ချပစ်မယ်
                # ဒါမှ စာတန်းတွေ ကပ်မနေဘဲ ကွက်လပ်လေး ပေါ်လာမှာပါ
                if end_time - start_time > 0.8:
                    end_time -= 0.15 

            start_ts = seconds_to_srt_time(start_time)
            end_ts = seconds_to_srt_time(end_time)
            
            block = f"{idx}\n{start_ts} --> {end_ts}\n{text}"
            f.write(block + "\n\n")
            f.flush()

            pct = min(95, int((seg.end / total_dur) * 100))
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["message"] = f"[{start_ts}] {text[:50]}..."
            idx += 1


def run_gemini_srt_translation(job_id: str, input_srt: str, output_srt: str, api_key: str, model_name: str, hint: str):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    with open(input_srt, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    total_blocks = len(blocks)
    batch_size = 60
    translated_batches: List[str] = []

    for start in range(0, total_blocks, batch_size):
        batch = blocks[start : start + batch_size]
        batch_text = "\n\n".join(batch)
        batch_num = (start // batch_size) + 1
        total_batches = (total_blocks + batch_size - 1) // batch_size

        JOBS[job_id]["message"] = f"Gemini ဖြင့် မြန်မာဘာသာပြန်နေပါသည် (Batch {batch_num}/{total_batches})..."
        JOBS[job_id]["progress"] = min(98, int((start / max(total_blocks, 1)) * 100))

        prompt = (
            "Translate the following .srt subtitle blocks into natural colloquial Burmese (Myanmar Unicode). "
            "Keep the exact index numbers and timestamps unchanged:\n\n"
            f"{batch_text}"
        )
        if hint:
            prompt = f"Movie Context: {hint}\n\n" + prompt

        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SRT_TRANSLATE_SYSTEM_PROMPT,
                        temperature=0.2,
                    ),
                )
                translated_batches.append(clean_llm_srt_output(response.text or ""))
                break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"Gemini API Error: {e}")
                time.sleep(3 * attempt)

        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_batches) + "\n")


def run_gemini_direct_video_to_mm(job_id: str, audio_path: str, work_dir: Path, output_srt: str, api_key: str, model_name: str, hint: str):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    chunk_seconds = 600
    pattern = str(work_dir / "chunk_%04d.mp3")

    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", pattern
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    chunk_files = sorted(glob.glob(str(work_dir / "chunk_*.mp3")))

    all_blocks: List[str] = []
    next_idx = 1

    for i, cpath in enumerate(chunk_files):
        offset_ms = i * chunk_seconds * 1000
        JOBS[job_id]["message"] = f"Gemini Direct Audio->MM (Chunk {i+1}/{len(chunk_files)})..."
        JOBS[job_id]["progress"] = int(((i) / max(len(chunk_files), 1)) * 95)

        uploaded = client.files.upload(file=cpath)
        try:
            while uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)

            prompt = (
                "Generate accurate Myanmar (Burmese Unicode) subtitles in standard .srt format "
                "for all spoken dialogue in this audio clip. Start timestamps from 00:00:00,000."
            )
            if hint:
                prompt += f"\nContext: {hint}"

            resp = client.models.generate_content(
                model=model_name,
                contents=[uploaded, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=AUDIO_TO_MM_SRT_SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            adj_blocks, next_idx = parse_and_offset_srt(resp.text or "", offset_ms, next_idx)
            all_blocks.extend(adj_blocks)

            with open(output_srt, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_blocks) + "\n")
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


# =====================================================================
# 3. BACKGROUND WORKER THREAD
# =====================================================================

def background_worker(job_id: str, params: Dict[str, Any]):
    work_dir = JOBS_DIR / job_id
    try:
        input_path = Path(params["input_path"])
        mode = params["mode"]
        whisper_model = params["whisper_model"]
        lang = params["lang"]
        gemini_model = params["gemini_model"]
        hint = params["hint"]
        api_key = params["api_key"]
        stem = params["orig_stem"]

        ext = input_path.suffix.lower()

        if ext == ".srt":
            mm_srt = str(work_dir / f"{stem}_mm.srt")
            run_gemini_srt_translation(job_id, str(input_path), mm_srt, api_key, gemini_model, hint)
            JOBS[job_id]["mm_srt"] = mm_srt
        else:
            JOBS[job_id]["message"] = "FFmpeg ဖြင့် အသံထုတ်ယူနေပါသည်..."
            audio_mp3 = str(work_dir / "extracted_audio.mp3")
            extract_lightweight_audio(str(input_path), audio_mp3)

            if mode == "original":
                orig_srt = str(work_dir / f"{stem}_orig.srt")
                run_whisper_original_srt(job_id, audio_mp3, orig_srt, whisper_model, lang)
                JOBS[job_id]["orig_srt"] = orig_srt

            elif mode == "both":
                orig_srt = str(work_dir / f"{stem}_orig.srt")
                mm_srt = str(work_dir / f"{stem}_mm.srt")
                run_whisper_original_srt(job_id, audio_mp3, orig_srt, whisper_model, lang)
                JOBS[job_id]["orig_srt"] = orig_srt
                run_gemini_srt_translation(job_id, orig_srt, mm_srt, api_key, gemini_model, hint)
                JOBS[job_id]["mm_srt"] = mm_srt

            elif mode == "myanmar":
                mm_srt = str(work_dir / f"{stem}_mm.srt")
                run_gemini_direct_video_to_mm(job_id, audio_mp3, work_dir, mm_srt, api_key, gemini_model, hint)
                JOBS[job_id]["mm_srt"] = mm_srt

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "စာတန်းထိုးဖိုင် ထုတ်ယူမှု အောင်မြင်စွာ ပြီးဆုံးပါပြီ!"

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"Error: {str(e)}"


# =====================================================================
# 4. WEB UI & REST API ENDPOINTS
# =====================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="my">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Subtitle Generator Pro (Render Paid Plan)</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen p-6">
  <div class="max-w-2xl mx-auto bg-slate-800 rounded-2xl shadow-xl p-8 border border-slate-700">
    <h1 class="text-2xl font-bold text-emerald-400 mb-2">🎬 Subtitle Generator Pro (Render Edition)</h1>
    <p class="text-sm text-slate-400 mb-6">
      မူရင်းဘာသာစကား (.srt) ကို API Key မလိုဘဲ Faster-Whisper ဖြင့် အခမဲ့ထုတ်ယူနိုင်ပြီး၊ မြန်မာဘာသာ (.srt) အတွက် Gemini API သုံးနိုင်ပါသည်။
    </p>

    <form id="uploadForm" class="space-y-4">
      <div>
        <label class="block text-sm font-medium mb-1">1. Video / Audio / .SRT ဖိုင် ရွေးပါ</label>
        <input type="file" name="file" required
          class="w-full text-sm bg-slate-900 border border-slate-700 rounded-lg p-2.5" />
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1">2. Output Mode</label>
          <select name="mode" id="modeSelect" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm">
            <option value="original">Original Language .SRT (API Key မလိုပါ)</option>
            <option value="both">Both (Original + Myanmar .SRT)</option>
            <option value="myanmar">Myanmar .SRT Only (Gemini Direct)</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">Whisper Model Size (Local RAM)</label>
          <select name="whisper_model" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm">
            <option value="base">base (Starter 512MB RAM အတွက်)</option>
            <option value="small" selected>small (Standard 2GB RAM အတွက် - အကြံပြု)</option>
            <option value="medium">medium (Pro 4GB+ RAM အတွက်)</option>
          </select>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1">Spoken Language Code (Optional)</label>
          <input type="text" name="lang" placeholder="e.g. en, zh, ko, ja (Auto if empty)"
            class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">Gemini API Key (မြန်မာဘာသာအတွက်သာ)</label>
          <input type="password" name="api_key" placeholder="Server ENV တွင်ထည့်ထားပါက မလိုပါ"
            class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
        </div>
      </div>

      <div>
        <label class="block text-sm font-medium mb-1">Movie Context Hint (Optional)</label>
        <input type="text" name="hint" placeholder="e.g. Chinese Wuxia drama, Action thriller"
          class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
      </div>

      <button type="submit" id="submitBtn"
        class="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold py-3 rounded-xl transition">
        🚀 Start Subtitle Generation
      </button>
    </form>

    <div id="statusBox" class="hidden mt-6 p-4 bg-slate-900 rounded-xl border border-slate-700">
      <div class="flex justify-between text-sm mb-2">
        <span id="statusText" class="text-emerald-400 font-medium">Processing...</span>
        <span id="progressPct" class="text-slate-400">0%</span>
      </div>
      <div class="w-full bg-slate-700 h-2.5 rounded-full overflow-hidden">
        <div id="progressBar" class="bg-emerald-400 h-2.5 w-0 transition-all duration-300"></div>
      </div>
      <div id="downloadLinks" class="mt-4 flex gap-3 hidden"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('uploadForm');
    const statusBox = document.getElementById('statusBox');
    const statusText = document.getElementById('statusText');
    const progressPct = document.getElementById('progressPct');
    const progressBar = document.getElementById('progressBar');
    const downloadLinks = document.getElementById('downloadLinks');
    const submitBtn = document.getElementById('submitBtn');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      submitBtn.disabled = true;
      statusBox.classList.remove('hidden');
      downloadLinks.classList.add('hidden');
      downloadLinks.innerHTML = '';
      statusText.textContent = 'Uploading file to Render server...';
      progressBar.style.width = '5%';

      const formData = new FormData(form);
      const res = await fetch('/api/start', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) {
        statusText.textContent = data.error || 'Upload failed';
        submitBtn.disabled = false;
        return;
      }

      const jobId = data.job_id;
      const timer = setInterval(async () => {
        const sRes = await fetch(`/api/status/${jobId}`);
        const sData = await sRes.json();

        statusText.textContent = sData.message;
        progressPct.textContent = `${sData.progress}%`;
        progressBar.style.width = `${sData.progress}%`;

        if (sData.status === 'completed') {
          clearInterval(timer);
          submitBtn.disabled = false;
          downloadLinks.classList.remove('hidden');
          if (sData.has_orig) {
            downloadLinks.innerHTML += `<a href="/api/download/${jobId}/orig" class="bg-sky-500 hover:bg-sky-600 text-slate-950 font-semibold px-4 py-2 rounded-lg text-sm">⬇️ Download Original .SRT</a>`;
          }
          if (sData.has_mm) {
            downloadLinks.innerHTML += `<a href="/api/download/${jobId}/mm" class="bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold px-4 py-2 rounded-lg text-sm">⬇️ Download Myanmar .SRT</a>`;
          }
        } else if (sData.status === 'failed') {
          clearInterval(timer);
          submitBtn.disabled = false;
        }
      }, 2000);
    });
  </script>
</body>
</html>
"""


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/start", methods=["POST"])
def start_job():
    if "file" not in request.files:
        return jsonify({"error": "ဖိုင်ရွေးချယ်ထားခြင်း မရှိပါ။"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "ဖိုင်အမည် မရှိပါ။"}), 400

    mode = request.form.get("mode", "original")
    whisper_model = request.form.get("whisper_model", os.environ.get("WHISPER_MODEL", "small"))
    lang = request.form.get("lang", "")
    hint = request.form.get("hint", "")
    gemini_model = request.form.get("gemini_model", "gemini-2.5-flash")
    api_key = request.form.get("api_key", "").strip() or os.environ.get("GEMINI_API_KEY", "")

    orig_path = Path(f.filename)
    if (mode in ["myanmar", "both"] or orig_path.suffix.lower() == ".srt") and not api_key:
        return jsonify({"error": "မြန်မာဘာသာပြန်ဆိုရန် GEMINI_API_KEY လိုအပ်ပါသည်။ (မူရင်းဘာသာသီးသန့်အတွက် Original Mode ကို ရွေးပါ)"}), 400

    job_id = uuid.uuid4().hex[:12]
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    saved_input = work_dir / f"input{orig_path.suffix.lower()}"
    f.save(str(saved_input))

    JOBS[job_id] = {
        "status": "running",
        "progress": 5,
        "message": "Queued on Render worker...",
        "orig_srt": None,
        "mm_srt": None,
    }

    params = {
        "input_path": str(saved_input),
        "orig_stem": orig_path.stem,
        "mode": mode,
        "whisper_model": whisper_model,
        "lang": lang,
        "gemini_model": gemini_model,
        "hint": hint,
        "api_key": api_key,
    }

    t = threading.Thread(target=background_worker, args=(job_id, params), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "has_orig": bool(job.get("orig_srt") and os.path.exists(job["orig_srt"])),
        "has_mm": bool(job.get("mm_srt") and os.path.exists(job["mm_srt"])),
    })


@app.route("/api/download/<job_id>/<kind>")
def download_srt(job_id: str, kind: str):
    job = JOBS.get(job_id)
    if not job:
        return "Job not found", 404

    target = job.get("orig_srt") if kind == "orig" else job.get("mm_srt")
    if not target or not os.path.exists(target):
        return "File not ready", 404

    return send_file(target, as_attachment=True, download_name=os.path.basename(target))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
