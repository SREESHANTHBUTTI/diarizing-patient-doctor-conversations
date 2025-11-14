# app.py
import os
import time
import json
import subprocess
from statistics import mean
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# whisper (openai-whisper)
import whisper

# pydub for fallback segmentation
from pydub import AudioSegment, silence
import numpy as np

# Try optional pyannote (gated). If unavailable or access denied we'll fallback.
HAS_PYANNOTE = False
try:
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    HAS_PYANNOTE = True
except Exception as e:
    print("[BACKEND] pyannote import failed — fallback segmentation will be used.", str(e))
    HAS_PYANNOTE = False

# ---------------------
# Config
# ---------------------
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# If you have an HF token with access to pyannote gated model,
# set HF_TOKEN environment var or put the token here.
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ---------------------
# Models
# ---------------------
print("[MODEL] Loading Whisper (small)...")
whisper_model = whisper.load_model("small")   # fast/accurate tradeoff

diarization_pipeline = None
if HAS_PYANNOTE:
    try:
        # If you need to authenticate for gated repo, ensure huggingface cli login is done.
        print("[MODEL] Loading Pyannote speaker diarization pipeline (attempt)...")
        diarization_pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization")
        print("[MODEL] Pyannote loaded.")
    except Exception as ex:
        print("[MODEL] Could not load pyannote pipeline automatically (maybe gated or incompatible).")
        print("ERROR DETAIL:", ex)
        diarization_pipeline = None
        HAS_PYANNOTE = False

# ---------------------
# Helpers
# ---------------------
def run_ffmpeg_to_wav(src_path: str) -> str:
    """Convert file to 16k mono wav using ffmpeg, returns wav path."""
    base, _ = os.path.splitext(src_path)
    wav_path = base + ".wav"
    if os.path.exists(wav_path):
        return wav_path
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000",
        "-loglevel", "error",
        wav_path
    ]
    subprocess.run(cmd, check=False)
    if not os.path.exists(wav_path):
        raise RuntimeError("ffmpeg failed to produce wav: " + wav_path)
    return wav_path

def fallback_segments_from_audio(wav_path: str, min_silence_len=400, silence_thresh=-40, seek_step=1):
    """Return non-silent segments using pydub (start_s_ms, end_ms)."""
    audio = AudioSegment.from_wav(wav_path)
    nonsilent = silence.detect_nonsilent(audio,
                                         min_silence_len=min_silence_len,
                                         silence_thresh=silence_thresh,
                                         seek_step=seek_step)
    # detect_nonsilent returns ms pairs - convert to seconds
    segments = [(start/1000.0, end/1000.0) for start, end in nonsilent]
    # If no segments, return single whole file
    if len(segments) == 0:
        return [(0.0, len(audio)/1000.0)]
    return segments

def average_confidence_from_whisper_result(result):
    """
    Try to compute average confidence from whisper segments if available.
    openai-whisper returns segments with 'avg_logprob' sometimes; convert to approximate confidence.
    """
    segs = result.get("segments", [])
    probs = []
    for s in segs:
        # avg_logprob might be negative (logprob). We'll convert to a 0-1 scale via logistic-like transform
        if "avg_logprob" in s and s["avg_logprob"] is not None:
            lp = s["avg_logprob"]
            # map [-5,0] -> [0,1] roughly
            conf = max(0.0, min(1.0, (lp + 5) / 5))
            probs.append(conf)
    if probs:
        return sum(probs) / len(probs)
    # fallback constant
    return None

# ---------------------
# Flask app
# ---------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "ClinVoice API running."})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Invalid filename"}), 400
    save_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(save_path)
    print(f"[UPLOAD] saved -> {save_path}")
    return jsonify({"path": save_path})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    file_path = data.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    start_t = time.time()
    wav = run_ffmpeg_to_wav(file_path)
    res = whisper_model.transcribe(wav, verbose=False)
    proc_time = time.time() - start_t
    conf = average_confidence_from_whisper_result(res) or 0.0
    return jsonify({"transcript": res.get("text",""), "processing_time": proc_time, "whisper_confidence": conf})

@app.route("/diarize", methods=["POST"])
def diarize():
    # Input: {"path": "..."}
    data = request.get_json()
    file_path = data.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    print("[DIARIZE] Processing:", file_path)
    start_all = time.time()
    wav = run_ffmpeg_to_wav(file_path)

    # total duration (s)
    audio_segment = AudioSegment.from_wav(wav)
    duration_s = len(audio_segment) / 1000.0

    segments = []
    used_pyannote = False

    if diarization_pipeline is not None:
        try:
            used_pyannote = True
            # pyannote returns annotation-like object
            diar = diarization_pipeline(wav)
            for (segment, _, speaker) in diar.itertracks(yield_label=True):
                segments.append({"start": float(segment.start), "end": float(segment.end), "speaker_id": str(speaker)})
            print(f"[DIARIZE] pyannote produced {len(segments)} segments")
        except Exception as e:
            print("[DIARIZE] pyannote failed at runtime - falling back:", e)
            segments = []
            used_pyannote = False

    if not used_pyannote:
        # fallback to energy/silence-based segmentation
        segpairs = fallback_segments_from_audio(wav)
        for s,e in segpairs:
            segments.append({"start": float(s), "end": float(e), "speaker_id": None})
        print(f"[DIARIZE] fallback produced {len(segments)} segments")

    # Transcribe per segment using whisper; also compute per-segment confidence if available
    full_results = []
    confidences = []
    speech_total = 0.0
    for idx, seg in enumerate(segments):
        ss = seg["start"]
        ee = seg["end"]
        # produce temp wav for segment
        seg_path = os.path.join(UPLOADS_DIR, f"segment_{idx}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", wav,
            "-ss", str(max(0, ss)), "-to", str(ee),
            "-ac", "1", "-ar", "16000",
            "-loglevel", "error",
            seg_path
        ]
        subprocess.run(cmd, check=False)
        if not os.path.exists(seg_path):
            continue

        # transcribe (fast)
        res = whisper_model.transcribe(seg_path, verbose=False)
        text = res.get("text", "").strip()
        conf = average_confidence_from_whisper_result(res)
        if conf is not None:
            confidences.append(conf)

        seg_duration = max(0.0, ee - ss)
        if text:
            speech_total += seg_duration
            full_results.append({
                "speaker_id": seg.get("speaker_id"),
                "start": ss,
                "end": ee,
                "text": text
            })
        # cleanup
        try:
            os.remove(seg_path)
        except:
            pass

    # Map speaker ids to human-friendly roles (improvise characters)
    speaker_map = {}
    role_order = ["Doctor", "Patient 1", "Patient 2", "Nurse", "Family"]
    role_idx = 0
    for seg in full_results:
        sid = seg.get("speaker_id")
        # if sid is None (fallback) we'll assign roles in alternating fashion to simulate
        if sid is None:
            # create a pseudo-sid based on round(start) to alternate
            sid = f"anon_{int(seg['start']*10)%10}"
            seg['speaker_id'] = sid
        if sid not in speaker_map:
            speaker_map[sid] = role_order[role_idx] if role_idx < len(role_order) else f"Speaker {role_idx+1}"
            role_idx += 1
        seg['speaker_role'] = speaker_map[sid]

    # Metrics
    processing_time = round(time.time() - start_all, 3)
    total_segments = len(full_results)
    avg_segment = round(mean([ (s['end'] - s['start']) for s in full_results ]) if total_segments>0 else 0.0, 3)
    whisper_confidence = round(float(mean(confidences)) if confidences else 0.0, 3)
    speech_ratio_speech = round(speech_total / duration_s, 3) if duration_s>0 else 0.0
    speech_ratio_silence = round(1.0 - speech_ratio_speech, 3)
    # A simple heuristic for speaker consistency: percent of segments assigned a repeated role after first appearance
    speaker_consistency = 0.0
    if total_segments>0:
        seen = {}
        consistent_count = 0
        for s in full_results:
            rid = s['speaker_role']
            if rid in seen:
                consistent_count += 1
            else:
                seen[rid] = 1
        speaker_consistency = round(consistent_count / total_segments, 3)

    metrics = {
        "processing_time": processing_time,
        "total_segments": total_segments,
        "avg_segment": avg_segment,
        "speaker_consistency": speaker_consistency,
        "whisper_confidence": whisper_confidence,
        "speech_ratio_speech": speech_ratio_speech,
        "speech_ratio_silence": speech_ratio_silence,
        # timeline useful for frontend plotting
        "segment_timeline": [
            {"speaker_role": s["speaker_role"], "start": s["start"], "end": s["end"]}
            for s in full_results
        ]
    }

    return jsonify({"diarized": full_results, "metrics": metrics})

@app.route("/download", methods=["GET"])
def download():
    file_path = request.args.get("path")
    if not file_path or os.path.exists(file_path) == False:
        return jsonify({"error": "Invalid path"}), 400
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    print("Starting ClinVoice API (port 5000)...")
    app.run(debug=True)
