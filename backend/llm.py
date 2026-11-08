"""Transcript -> structured clinical record, using Llama 3.3 70B on Groq's free tier."""
import json
import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a clinical documentation assistant. You receive the raw transcript of a
doctor-patient consultation (no speaker labels, may contain recognition errors).

Extract ONLY what is explicitly stated. Never invent, infer, or add medical knowledge.
- If something is not mentioned, use null (or [] for lists).
- Keep the patient's own wording for chief_complaint.
- symptoms.positive = symptoms the patient has; symptoms.negative = symptoms explicitly denied.
- clinical_observations = vitals/exam findings stated aloud. Keep numbers and units exactly.
- assessment = diagnosis or differentials as stated by the clinician only. If none stated, null.
- plan = investigations, prescriptions, advice, follow-up as stated.
- clinical_summary = 3-5 sentences, professional, suitable for a medical record, using only extracted facts.
Speech-recognition errors: the transcript comes from automatic speech recognition and may contain
misheard words, especially medical terms and drug names (e.g. "hyper tension" for "hypertension").
- If a word is clearly a mis-hearing of a medical term or drug name, judging by sound AND context,
  use the correct term in the extracted fields and record it in transcription_notes.corrected.
- If a word looks wrong but you cannot confidently tell what was meant, do NOT guess. Keep the
  original wording and add it to transcription_notes.unclear.
- A relieving factor is only something the patient says helps. Feeling the need to lie down is not relief.
- Do not add words that were not said. If the patient says "back pain", do not write "low back pain".
- transcription_notes.corrected[].heard must be copied exactly from the transcript. If you cannot
  point to the exact misheard word in the transcript, do not add an entry.
- Record past or stopped treatments (drugs and therapies such as physical therapy) in
  medication_history.past_treatments, with the reason they were stopped if stated.
Return a single JSON object with exactly this shape:
{
  "patient_details": {"name": str|null, "age": str|null, "sex": str|null, "identifiers": [str]},
  "chief_complaint": str|null,
  "history_of_present_illness": {"onset": str|null, "duration": str|null, "progression": str|null,
                                  "aggravating_factors": [str], "relieving_factors": [str], "other": str|null},
  "symptoms": {"positive": [str], "negative": [str]},
  "past_medical_history": {"conditions": [str], "surgeries": [str], "hospitalisations": [str]},
  "medication_history": {"current_medications": [{"name": str, "dose": str|null, "frequency": str|null}],
                          "adherence": str|null, "allergies": [str]},"past_treatments": [{"name": str, "status": str|null}],
  "clinical_observations": {"vitals": [str], "examination_findings": [str]},
  "assessment": {"provisional_diagnosis": str|null, "differentials": [str]},
  "plan": {"investigations": [str], "prescriptions": [{"name": str, "dose": str|null, "frequency": str|null, "duration": str|null}],
           "advice": [str], "follow_up": str|null},
  "transcription_notes": {"corrected": [{"heard": str, "corrected_to": str}], "unclear": [str]},
       
  "clinical_summary": str
}"""


class LLMError(Exception):
    pass


def analyze(transcript: str) -> dict:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set. Add it to the .env file.")

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n\"\"\"\n{transcript}\n\"\"\""},
            ],
        },
        timeout=90,
    )
    if resp.status_code != 200:
        raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    try:
        data = json.loads(resp.json()["choices"][0]["message"]["content"])
    except (KeyError, ValueError) as e:
        raise LLMError(f"Could not parse model output: {e}")

    # Drop "corrections" whose misheard word isn't actually in the transcript
    notes = data.get("transcription_notes") or {}
    low = transcript.lower()
    notes["corrected"] = [
        c for c in (notes.get("corrected") or [])
        if isinstance(c, dict) and c.get("heard") and c["heard"].lower() in low
    ]
    data["transcription_notes"] = notes
    return data