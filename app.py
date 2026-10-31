#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render Paid Plan - Subtitle Generator Web Service v2 (app-v2.py)
================================================================
Supports:
1. Groq API (whisper-large-v3-turbo / whisper-large-v3) -> Ultra-fast Original .SRT (Low RAM on Render!)
2. Local Faster-Whisper (No API Key)                    -> Free Original .SRT on CPU
3. Groq LLM (llama-3.3-70b-versatile)                   -> Myanmar .SRT Translation using Groq API Key
4. Gemini API (gemini-2.5-flash)                        -> Natural Myanmar .SRT Translation
"""

import os
import re
import uuid
import time
import glob
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

JOBS_DIR = Path(tempfile.gettempdir()) / "render_sub_jobs_v2"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}
WHISPER_CACHE: Dict[str, Any] = {}
WHISPER_LOCK = threading.Lock()

# =====================================================================
# 1. SYSTEM PROMPTS FOR MYANMAR TRANSLATION (GROQ & GEMINI)
# =====================================================================

SRT_TRANSLATE_SYSTEM_PROMPT = """# Role & Objective
You are a professional Movie Subtitle Translator specializing in translating foreign `.srt` subtitles into **natural, colloquial Burmese (Myanmar Unicode)**.

# Core Rules (CRITICAL)
1. Keep the exact subtitle index numbers and timestamps (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) unchanged.
2. Translate ONLY the dialogue text into natural spoken Burmese suited to the character's tone (e.g., ကျွန်တော်/ခင်ဗျား, ငါ/မင်း, ရှင်/ကျွန်မ).
3. Return ONLY the valid `.srt` blocks without markdown code fences or extra commentary.
4. Use 100% standard Myanmar Unicode (NOT Zawgyi).
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
    text = re.sub(r"^```(?:srt|plaintext|text)?\s*
?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"
?```\s*$", "", text)
    return text.strip()


# =====================================================================
# 2. FFMPEG AUDIO EXTRACTION & CHUNKING (FOR GROQ 25MB LIMIT & LOCAL)
# =====================================================================

def extract_lightweight_audio(input_media: str, output_mp3: str):
    """Extract 16kHz Mono 64kbps MP3 (~4.8MB per 10 mins) via FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_media,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        output_mp3
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")


def split_audio_into_chunks(audio_mp3: str, work_dir: Path, chunk_seconds: int = 600) -> List[Tuple[str, int]]:
    """Split extracted MP3 into 10-minute chunks (~4.8MB each) so Groq API never hits the 25MB limit."""
    pattern = str(work_dir / "chunk_%04d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", audio_mp3,
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", pattern
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg segmenting failed: {res.stderr}")

    chunk_files = sorted(glob.glob(str(work_dir / "chunk_*.mp3")))
    return [(fpath, idx * chunk_seconds * 1000) for idx, fpath in enumerate(chunk_files)]


# =====================================================================
# 3. GROQ WHISPER API (ULTRA-FAST ORIGINAL .SRT GENERATION)
# =====================================================================

def run_groq_whisper_original_srt(
    job_id: str,
    audio_mp3: str,
    work_dir: Path,
    output_srt: str,
    groq_api_key: str,
    groq_whisper_model: str = "whisper-large-v3-turbo",
    lang_code: str = ""
):
    """
    Use Groq Cloud Whisper API (whisper-large-v3-turbo or whisper-large-v3)
    to transcribe audio chunks into frame-accurate Original Language .srt.
    """
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    chunks = split_audio_into_chunks(audio_mp3, work_dir, chunk_seconds=600)

    all_blocks: List[str] = []
    idx = 1

    for c_i, (chunk_path, offset_ms) in enumerate(chunks, start=1):
        JOBS[job_id]["message"] = f"Groq {groq_whisper_model} ဖြင့် မူရင်းစာတန်းထိုး ထုတ်နေပါသည် (Chunk {c_i}/{len(chunks)})..."
        JOBS[job_id]["progress"] = min(90, int(((c_i - 1) / max(len(chunks), 1)) * 90) + 10)

        for attempt in range(1, 4):
            try:
                with open(chunk_path, "rb") as audio_file:
                    kwargs: Dict[str, Any] = {
                        "file": (os.path.basename(chunk_path), audio_file.read()),
                        "model": groq_whisper_model,
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["segment"],
                        "temperature": 0.0,
                    }
                    if lang_code and lang_code.strip():
                        kwargs["language"] = lang_code.strip()

                    transcription = client.audio.transcriptions.create(**kwargs)

                segments = getattr(transcription, "segments", None) or []
                for seg in segments:
                    seg_start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
                    seg_end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
                    seg_text = (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip()

                    if not seg_text:
                        continue

                    start_ms = int(round(float(seg_start) * 1000)) + offset_ms
                    end_ms = int(round(float(seg_end) * 1000)) + offset_ms

                    block = f"{idx}
{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}
{seg_text}"
                    all_blocks.append(block)
                    idx += 1

                # Save intermediate progress after each chunk
                with open(output_srt, "w", encoding="utf-8") as f:
                    f.write("

".join(all_blocks) + "
")
                break

            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"Groq Whisper API Error on chunk {c_i}: {e}")
                time.sleep(3 * attempt)


# =====================================================================
# 4. LOCAL FASTER-WHISPER (NO API KEY FALLBACK)
# =====================================================================

def get_local_whisper_model(model_size: str = "small"):
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


def run_local_whisper_original_srt(job_id: str, audio_path: str, output_srt: str, model_size: str, lang_code: str):
    JOBS[job_id]["message"] = f"Loading Local Faster-Whisper ({model_size}, int8)..."
    model = get_local_whisper_model(model_size)

    JOBS[job_id]["message"] = "Local Faster-Whisper ဖြင့် မူရင်းဘာသာ .srt ထုတ်နေပါသည်..."
    language_arg = lang_code.strip() if lang_code and lang_code.strip() else None

    segments, info = model.transcribe(
        audio_path,
        language=language_arg,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    total_dur = max(info.duration, 1.0)
    idx = 1

    with open(output_srt, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            start_ts = seconds_to_srt_time(seg.start)
            end_ts = seconds_to_srt_time(seg.end)
            block = f"{idx}
{start_ts} --> {end_ts}
{text}"
            f.write(block + "

")
            f.flush()

            pct = min(95, int((seg.end / total_dur) * 100))
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["message"] = f"[{start_ts}] {text[:50]}..."
            idx += 1


# =====================================================================
# 5. MYANMAR .SRT TRANSLATION (GROQ LLAMA-3.3-70B OR GEMINI)
# =====================================================================

def translate_srt_with_groq(
    job_id: str,
    input_srt: str,
    output_srt: str,
    groq_api_key: str,
    groq_llm_model: str = "llama-3.3-70b-versatile",
    hint: str = ""
):
    """Translate Original .srt blocks into Myanmar Unicode using Groq Chat Completions."""
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    with open(input_srt, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    blocks = re.split(r"
\s*
", content)
    total_blocks = len(blocks)
    batch_size = 40  # Smaller batch size for Groq token limits and faster responses
    translated_batches: List[str] = []

    for start in range(0, total_blocks, batch_size):
        batch = blocks[start : start + batch_size]
        batch_text = "

".join(batch)
        batch_num = (start // batch_size) + 1
        total_batches = (total_blocks + batch_size - 1) // batch_size

        JOBS[job_id]["message"] = f"Groq ({groq_llm_model}) ဖြင့် မြန်မာဘာသာပြန်နေပါသည် (Batch {batch_num}/{total_batches})..."
        JOBS[job_id]["progress"] = min(98, int((start / max(total_blocks, 1)) * 100))

        user_prompt = (
            "Translate the following .srt subtitle blocks into natural colloquial Burmese (Myanmar Unicode). "
            "Keep the exact index numbers and timestamps unchanged:

"
            f"{batch_text}"
        )
        if hint:
            user_prompt = f"Movie Context: {hint}

" + user_prompt

        for attempt in range(1, 4):
            try:
                completion = client.chat.completions.create(
                    model=groq_llm_model,
                    messages=[
                        {"role": "system", "content": SRT_TRANSLATE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                reply = completion.choices[0].message.content or ""
                translated_batches.append(clean_llm_srt_output(reply))
                break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"Groq Translation Error: {e}")
                time.sleep(3 * attempt)

        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("

".join(translated_batches) + "
")


def translate_srt_with_gemini(
    job_id: str,
    input_srt: str,
    output_srt: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.5-flash",
    hint: str = ""
):
    """Translate Original .srt blocks into Myanmar Unicode using Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key)
    with open(input_srt, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    blocks = re.split(r"
\s*
", content)
    total_blocks = len(blocks)
    batch_size = 60
    translated_batches: List[str] = []

    for start in range(0, total_blocks, batch_size):
        batch = blocks[start : start + batch_size]
        batch_text = "

".join(batch)
        batch_num = (start // batch_size) + 1
        total_batches = (total_blocks + batch_size - 1) // batch_size

        JOBS[job_id]["message"] = f"Gemini ({gemini_model}) ဖြင့် မြန်မာဘာသာပြန်နေပါသည် (Batch {batch_num}/{total_batches})..."
        JOBS[job_id]["progress"] = min(98, int((start / max(total_blocks, 1)) * 100))

        prompt = (
            "Translate the following .srt subtitle blocks into natural colloquial Burmese (Myanmar Unicode). "
            "Keep the exact index numbers and timestamps unchanged:

"
            f"{batch_text}"
        )
        if hint:
            prompt = f"Movie Context: {hint}

" + prompt

        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=gemini_model,
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
            f.write("

".join(translated_batches) + "
")


# =====================================================================
# 6. BACKGROUND WORKER THREAD
# =====================================================================

def background_worker(job_id: str, params: Dict[str, Any]):
    work_dir = JOBS_DIR / job_id
    try:
        input_path = Path(params["input_path"])
        mode = params["mode"]                     # "original", "both", "myanmar"
        stt_engine = params["stt_engine"]         # "groq_whisper" or "local_whisper"
        translate_engine = params["translate_engine"]  # "groq_llm" or "gemini"
        lang = params["lang"]
        hint = params["hint"]
        groq_api_key = params["groq_api_key"]
        gemini_api_key = params["gemini_api_key"]
        stem = params["orig_stem"]

        ext = input_path.suffix.lower()

        # Case 1: Uploaded file is already an .srt -> Translate to Myanmar
        if ext == ".srt":
            mm_srt = str(work_dir / f"{stem}_mm.srt")
            if translate_engine == "groq_llm":
                translate_srt_with_groq(job_id, str(input_path), mm_srt, groq_api_key, "llama-3.3-70b-versatile", hint)
            else:
                translate_srt_with_gemini(job_id, str(input_path), mm_srt, gemini_api_key, "gemini-2.5-flash", hint)
            JOBS[job_id]["mm_srt"] = mm_srt

        # Case 2: Video/Audio input
        else:
            JOBS[job_id]["message"] = "FFmpeg ဖြင့် အသံထုတ်ယူနေပါသည်..."
            audio_mp3 = str(work_dir / "extracted_audio.mp3")
            extract_lightweight_audio(str(input_path), audio_mp3)

            orig_srt = str(work_dir / f"{stem}_orig.srt")

            # Step 1: Transcribe Original Language .SRT (Groq Whisper OR Local Faster-Whisper)
            if stt_engine == "groq_whisper":
                run_groq_whisper_original_srt(
                    job_id=job_id,
                    audio_mp3=audio_mp3,
                    work_dir=work_dir,
                    output_srt=orig_srt,
                    groq_api_key=groq_api_key,
                    groq_whisper_model="whisper-large-v3-turbo",
                    lang_code=lang,
                )
            else:
                run_local_whisper_original_srt(
                    job_id=job_id,
                    audio_path=audio_mp3,
                    output_srt=orig_srt,
                    model_size=os.environ.get("WHISPER_MODEL", "small"),
                    lang_code=lang,
                )

            if mode in ["original", "both"]:
                JOBS[job_id]["orig_srt"] = orig_srt

            # Step 2: If Myanmar output is requested, translate the generated Original .SRT
            if mode in ["both", "myanmar"]:
                mm_srt = str(work_dir / f"{stem}_mm.srt")
                if translate_engine == "groq_llm":
                    translate_srt_with_groq(job_id, orig_srt, mm_srt, groq_api_key, "llama-3.3-70b-versatile", hint)
                else:
                    translate_srt_with_gemini(job_id, orig_srt, mm_srt, gemini_api_key, "gemini-2.5-flash", hint)
                JOBS[job_id]["mm_srt"] = mm_srt

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "စာတန်းထိုးဖိုင် ထုတ်ယူမှု အောင်မြင်စွာ ပြီးဆုံးပါပြီ!"

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"Error: {str(e)}"


# =====================================================================
# 7. WEB UI & REST API ENDPOINTS
# =====================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="my">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Subtitle Generator Pro v2 (Groq + Gemini + Render)</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen p-6">
  <div class="max-w-2xl mx-auto bg-slate-800 rounded-2xl shadow-xl p-8 border border-slate-700">
    <h1 class="text-2xl font-bold text-emerald-400 mb-2">⚡ Subtitle Generator v2 (Groq & Gemini Edition)</h1>
    <p class="text-sm text-slate-400 mb-6">
      Groq API (<code>whisper-large-v3-turbo</code> & <code>llama-3.3-70b</code>)၊ Local Whisper နှင့် Gemini API တို့ကို စိတ်ကြိုက်တွဲဖက်သုံးနိုင်ပါသည်။
    </p>

    <form id="uploadForm" class="space-y-4">
      <div>
        <label class="block text-sm font-medium mb-1">1. Video / Audio / .SRT ဖိုင် ရွေးပါ</label>
        <input type="file" name="file" required
          class="w-full text-sm bg-slate-900 border border-slate-700 rounded-lg p-2.5" />
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1">2. Output Mode</label>
          <select name="mode" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm">
            <option value="original">Original .SRT သီးသန့်</option>
            <option value="both" selected>Both (Original + Myanmar .SRT)</option>
            <option value="myanmar">Myanmar .SRT သီးသန့်</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">3. Audio -> Original Engine</label>
          <select name="stt_engine" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm">
            <option value="groq_whisper" selected>⚡ Groq Whisper Turbo (အမြန်ဆုံး)</option>
            <option value="local_whisper">💻 Local Faster-Whisper (No API Key)</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">4. Myanmar Translator</label>
          <select name="translate_engine" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm">
            <option value="groq_llm">⚡ Groq (Llama-3.3-70B)</option>
            <option value="gemini" selected>✨ Gemini 2.5 Flash (မြန်မာစာ အကောင်းဆုံး)</option>
          </select>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1">Groq API Key (gsk_...)</label>
          <input type="password" name="groq_api_key" placeholder="Server ENV တွင်ထည့်ထားပါက မလိုပါ"
            class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">Gemini API Key (Optional)</label>
          <input type="password" name="gemini_api_key" placeholder="Gemini သုံးမှသာ လိုအပ်ပါသည်"
            class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1">Spoken Language Code (Optional)</label>
          <input type="text" name="lang" placeholder="e.g. en, zh, ko, ja (Auto if empty)"
            class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">Movie Context Hint (Optional)</label>
          <input type="text" name="hint" placeholder="e.g. Chinese Wuxia drama, Action thriller"
            class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm" />
        </div>
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

    mode = request.form.get("mode", "both")
    stt_engine = request.form.get("stt_engine", "groq_whisper")
    translate_engine = request.form.get("translate_engine", "groq_llm")
    lang = request.form.get("lang", "")
    hint = request.form.get("hint", "")

    groq_api_key = request.form.get("groq_api_key", "").strip() or os.environ.get("GROQ_API_KEY", "")
    gemini_api_key = request.form.get("gemini_api_key", "").strip() or os.environ.get("GEMINI_API_KEY", "")

    orig_path = Path(f.filename)
    is_srt = orig_path.suffix.lower() == ".srt"

    if not is_srt and stt_engine == "groq_whisper" and not groq_api_key:
        return jsonify({"error": "Groq Whisper သုံးရန် GROQ_API_KEY လိုအပ်ပါသည်။"}), 400

    if (mode in ["both", "myanmar"] or is_srt):
        if translate_engine == "groq_llm" and not groq_api_key:
            return jsonify({"error": "Groq ဖြင့် မြန်မာဘာသာပြန်ရန် GROQ_API_KEY လိုအပ်ပါသည်။"}), 400
        if translate_engine == "gemini" and not gemini_api_key:
            return jsonify({"error": "Gemini ဖြင့် မြန်မာဘာသာပြန်ရန် GEMINI_API_KEY လိုအပ်ပါသည်။"}), 400

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
        "stt_engine": stt_engine,
        "translate_engine": translate_engine,
        "lang": lang,
        "hint": hint,
        "groq_api_key": groq_api_key,
        "gemini_api_key": gemini_api_key,
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
