# MedScribe

Live consultation transcription with an AI-generated clinical summary.

Speak into the microphone and the transcript appears on screen as you talk. When the consultation ends, the full transcript is analysed by an LLM and turned into a structured clinical summary.

> **Demo only.** Not a medical device. Output is a draft that a clinician must review. Use dummy consultations, never real patient data, because the transcript is sent to a third-party API.

## How it works

```
Browser                                 Local FastAPI server
───────                                 ────────────────────
Microphone
  │
Silero VAD (drops silence/noise)
  │  speech segments
  ▼
POST /api/transcribe  ───────────────▶  NVIDIA Parakeet TDT 0.6B v2 (CPU, ONNX)
  ◀── text ───────────────────────────
Live transcript panel

"Process Transcript with AI"
POST /api/analyze  ──────────────────▶  Groq API: openai/gpt-oss-120b
  ◀── structured JSON ────────────────
Missing name/age/gender? → prompt form
AI Medical Summary panel
```

| Stage | Technology | Runs on |
|---|---|---|
| Audio capture + Voice Activity Detection | Silero VAD (`@ricky0123/vad-web`) | Browser |
| Speech recognition | `nvidia/parakeet-tdt-0.6b-v2` via `onnx-asr` | Your CPU (local) |
| Analysis and summary | `openai/gpt-oss-120b` on Groq (free tier) | Groq cloud |
| Backend | Python, FastAPI | Local |
| Frontend | HTML, CSS, JavaScript | Browser |

## Features

- Live transcript with word and chunk counters
- VAD so silence and background noise never reach the speech model
- Structured extraction: patient details, chief complaint, history of present illness, symptoms (positive and denied kept separate), past medical history, medications (current and past), examination findings, assessment, plan
- Asks for name, age and gender if they were not said in the conversation
- **Transcription Check**: lists medical terms the AI corrected after likely mishearing, and words it was unsure about. Corrections are verified against the transcript in code.
- Copy summary as plain text, save the recording as WAV

## Project structure

```
medscribe/
├── backend/
│   ├── main.py      # FastAPI app and API routes
│   ├── asr.py       # Parakeet speech recognition
│   └── llm.py       # Prompt, Groq call, output checks
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js       # Mic, VAD, live transcript, summary rendering
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

**Requirements:** Python 3.10–3.12, Chrome or Edge, internet (for the first model download, the VAD files and the LLM).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
copy .env.example .env            # macOS / Linux: cp .env.example .env
# Edit .env and add your free key from https://console.groq.com/keys

# 4. Run
uvicorn backend.main:app --reload
```

The first start downloads the Parakeet model (about 600 MB, one time). Wait for `Parakeet model ready.` in the terminal, then open **http://localhost:8000** in Chrome. Use `localhost`, because the browser only allows microphone access on secure origins.

## Configuration (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key | required |
| `GROQ_MODEL` | LLM used for analysis | `openai/gpt-oss-120b` |
| `PARAKEET_QUANT` | `int8` = faster, slightly less accurate. Empty = full precision, more accurate. | empty |

Restart the server after changing `.env`.

## Usage

1. Click **Start Mic** and allow microphone access.
2. Speak (or play a recorded consultation). Text appears after each pause.
3. Click **Stop**.
4. Click **Process Transcript with AI**. If name, age or gender were not said, a form asks for them.
5. Review the summary and the **Transcription Check** section.
6. **Copy to Prescription** copies a plain-text version.

## Improving accuracy

- Use a good microphone in a quiet room, one speaker at a time.
- Keep `PARAKEET_QUANT` empty for best accuracy.
- Tune the VAD in `frontend/app.js` (`redemptionMs`, `preSpeechPadMs`, `minSpeechMs`, thresholds). A shorter `redemptionMs` gives more frequent live updates. A longer one keeps sentences together.

## Troubleshooting

| Problem | Fix |
|---|---|
| `model_not_found` from Groq | The model was retired or renamed. Check Groq's [deprecations page](https://console.groq.com/docs/deprecations) and set `GROQ_MODEL` in `.env`. |
| `GROQ_API_KEY is not set` | `.env` must be in the project root, not in `backend/`. Restart the server. |
| Page won't load | The server must be running. Don't open `index.html` by double-clicking it. |
| Mic does nothing | Open via `localhost`, allow the mic (lock icon in the address bar), and make sure you are online (VAD files load from a CDN). |
| UI changes don't show | Hard refresh with Ctrl+Shift+R. |
| `pip install` fails on `onnx-asr` | Check `python --version` is 3.10–3.12. |

## Known limitations

- No speaker labels. The LLM infers who said what from context.
- Speech recognition can mishear medical terms and drug names. The AI corrects only clear cases and flags the rest, but clinician review is still required.
- The free LLM tier has rate limits, and models can be retired by the provider.
- Transcripts leave your machine at the analysis step. Do not use real patient data.

## Security

Never commit `.env`. If a key is ever exposed, delete it in the Groq console and create a new one.