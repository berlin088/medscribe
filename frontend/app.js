// MedScribe frontend: mic -> Silero VAD -> /api/transcribe (Parakeet) -> live transcript
// On "Process", full transcript -> /api/analyze (Llama 3.3 70B) -> structured summary.

const $ = (id) => document.getElementById(id);

const state = {
  vad: null,
  listening: false,
  segments: [],        // Float32Array per speech segment (for Save Recording)
  lines: [],           // transcript text per segment, in order
  queue: Promise.resolve(), // keeps transcription results in speaking order
  pending: 0,
  record: null,        // last structured result
};

// ---------- Audio capture + VAD ----------
async function startMic() {
  if (!state.vad) {
    setStatus("Loading voice-activity model…");
    state.vad = await vad.MicVAD.new({
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      redemptionMs: 1000,     // wait 1 s of silence before closing a chunk
      preSpeechPadMs: 400,    // keep the start of the first word
      minSpeechMs: 400,
      baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/",
      onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
      onSpeechStart: () => setStatus("🔴 Speech detected…"),
      onVADMisfire: () => setStatus("Listening… (ignored a short noise)"),
      onSpeechEnd: (audio) => handleSegment(audio), // Float32Array, 16 kHz mono
    });
  }
  state.vad.start();
  state.listening = true;
  $("btnStart").disabled = true;
  $("btnStart").classList.add("listening");
  $("btnStop").disabled = false;
  setStatus("Listening…");
}

function stopMic() {
  if (state.vad) state.vad.pause();
  state.listening = false;
  $("btnStart").disabled = false;
  $("btnStart").classList.remove("listening");
  $("btnStop").disabled = true;
  setStatus(state.pending ? "Finishing transcription…" : "Stopped.");
}

// ---------- ASR ----------
function handleSegment(audio) {
  state.segments.push(audio);
  state.pending++;
  setStatus("Transcribing…");
  // Chain so text appears in the order it was spoken even if responses return unevenly.
  state.queue = state.queue.then(async () => {
    try {
      const res = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: audio.buffer.slice(audio.byteOffset, audio.byteOffset + audio.byteLength),
      });
      if (!res.ok) throw new Error(await res.text());
      const { text } = await res.json();
      if (text) { state.lines.push(text); renderTranscript(); }
    } catch (e) {
      showError("Transcription failed: " + e.message);
    } finally {
      state.pending--;
      if (state.listening) setStatus("Listening…");
      else if (!state.pending) setStatus("Stopped.");
    }
  });
}

function renderTranscript() {
  const box = $("transcript");
  box.innerHTML = "";
  if (!state.lines.length) {
    box.innerHTML = '<span class="placeholder">Ready — click Start Mic and speak naturally. VAD will auto-chunk your speech.</span>';
  } else {
    state.lines.forEach((t) => { const p = document.createElement("p"); p.textContent = t; box.appendChild(p); });
    box.scrollTop = box.scrollHeight;
  }
  const words = state.lines.join(" ").split(/\s+/).filter(Boolean).length;
  $("words").textContent = words;
  $("chunks").textContent = state.lines.length;
}

function clearTranscript() {
  state.lines = []; state.segments = [];
  renderTranscript();
}

// ---------- LLM analysis ----------
async function processTranscript() {
  const transcript = state.lines.join("\n").trim();
  if (!transcript) return showError("Nothing to process yet. Record something first.");
  if (state.listening) stopMic();
  await state.queue; // wait for in-flight chunks
  hideError();
  const btn = $("btnProcess");
  btn.disabled = true; btn.textContent = "Analysing…";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript: state.lines.join("\n") }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    state.record = await res.json();
    await askMissingPatientDetails(state.record); // pause here if name/age/sex weren't said
    renderRecord(state.record);
  } catch (e) {
    showError("Analysis failed: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "⚡ Process Transcript with AI";
  }
}

// ---------- Ask for patient details the conversation didn't contain ----------
const MISSING = /^\s*(unknown|not (mentioned|stated|specified|provided)|n\/a|na|none|null|-+)?\s*$/i;
const isMissing = (v) => v == null || MISSING.test(String(v));

function askMissingPatientDetails(record) {
  const pd = (record.patient_details = record.patient_details || {});
  const keys = ["name", "age", "sex"];
  const need = keys.filter((k) => isMissing(pd[k]));
  if (!need.length) return Promise.resolve();

  return new Promise((resolve) => {
    const modal = $("patientModal");
    keys.forEach((k) => {
      $("pm-" + k + "-wrap").hidden = !need.includes(k);
      $("pm-" + k).value = "";
    });
    modal.hidden = false;
    $("pm-" + need[0]).focus();

    const close = () => { modal.hidden = true; resolve(); };
    $("patientForm").onsubmit = (e) => {
      e.preventDefault();
      need.forEach((k) => { pd[k] = $("pm-" + k).value.trim() || null; });
      close();
    };
    $("pm-skip").onclick = () => { need.forEach((k) => { pd[k] = null; }); close(); };
  });
}

// ---------- Render structured record ----------
const arr = (x) => (Array.isArray(x) ? x.filter(Boolean) : []);
const fmtDose = (m) => [m.name, m.dose, m.frequency, m.duration].filter(Boolean).join(" · ");

function fill(id, items, emptyText) {
  const el = $(id);
  el.className = items.length ? "" : "empty";
  el.innerHTML = "";
  if (!items.length) { el.textContent = emptyText; return; }
  const ul = document.createElement("ul"); ul.className = "filled-list";
  items.forEach((it) => {
    const li = document.createElement("li");
    if (typeof it === "string") li.textContent = it;
    else { li.className = it.cls || ""; li.textContent = it.text; }
    ul.appendChild(li);
  });
  el.appendChild(ul);
}

function renderRecord(r) {
  const pd = r.patient_details || {};
  $("pName").textContent = pd.name || "—";
  $("pAge").textContent = pd.age || "—";
  $("pSex").textContent = pd.sex || "—";

  const cs = $("sClinical");
  cs.textContent = r.clinical_summary || "No summary generated";
  cs.classList.toggle("filled", !!r.clinical_summary);

  fill("sChief", r.chief_complaint ? [r.chief_complaint] : [], "No complaints extracted yet");

  const h = r.history_of_present_illness || {};
  fill("sHpi", [
    h.onset && "Onset: " + h.onset,
    h.duration && "Duration: " + h.duration,
    h.progression && "Progression: " + h.progression,
    arr(h.aggravating_factors).length && "Aggravating: " + arr(h.aggravating_factors).join(", "),
    arr(h.relieving_factors).length && "Relieving: " + arr(h.relieving_factors).join(", "),
    h.other,
  ].filter(Boolean), "No history extracted yet");

  const s = r.symptoms || {};
  fill("sSymptoms", [
    ...arr(s.positive),
    ...arr(s.negative).map((t) => ({ text: "Denies: " + t, cls: "neg" })),
  ], "No symptoms extracted yet");

  const a = r.assessment || {};
  fill("sAssessment", [
    a.provisional_diagnosis && "Provisional: " + a.provisional_diagnosis,
    ...arr(a.differentials).map((d) => "Differential: " + d),
  ].filter(Boolean), "No diagnosis stated");

  const mh = r.medication_history || {};
  fill("sMeds", [
    ...arr(mh.current_medications).map(fmtDose),
    mh.adherence && "Adherence: " + mh.adherence,
  ].filter(Boolean), "No medications extracted yet");

  const co = r.clinical_observations || {};
  fill("sExam", [...arr(co.vitals), ...arr(co.examination_findings)], "No examinations extracted yet");

  fill("sAllergies", arr(mh.allergies), "No allergies extracted yet");
  fill("sMeds", [
    ...arr(mh.current_medications).map(fmtDose),
    ...arr(mh.past_treatments).map((t) => "Past: " + [t.name, t.status].filter(Boolean).join(" — ")),
    mh.adherence && "Adherence: " + mh.adherence,
  ].filter(Boolean), "No medications extracted yet");

  const p = r.past_medical_history || {};
  fill("sPmh", [
    ...arr(p.conditions),
    ...arr(p.surgeries).map((x) => "Surgery: " + x),
    ...arr(p.hospitalisations).map((x) => "Hospitalised: " + x),
  ], "No past medical history extracted");

  const pl = r.plan || {};
  fill("sPlan", [
    ...arr(pl.investigations).map((x) => "Investigation: " + x),
    ...arr(pl.prescriptions).map((x) => "Rx: " + fmtDose(x)),
    ...arr(pl.advice).map((x) => "Advice: " + x),
    pl.follow_up && "Follow-up: " + pl.follow_up,
  ].filter(Boolean), "No plan extracted yet");
  const tn = r.transcription_notes || {};
  fill("sNotes", [
  ...arr(tn.corrected).map((c) => `Heard "${c.heard}" → read as "${c.corrected_to}"`),
  ...arr(tn.unclear).map((u) => "Unclear: " + u),
], "Nothing flagged");

}

function clearSummary() {
  state.record = null;
  ["pName", "pAge", "pSex"].forEach((i) => ($(i).textContent = "—"));
  $("sClinical").textContent = "AI-generated clinical summary will appear here…";
  $("sClinical").classList.remove("filled");
  const defaults = {
    sChief: "No complaints extracted yet", sHpi: "No history extracted yet",
    sSymptoms: "No symptoms extracted yet", sAssessment: "No diagnosis inferred",
    sMeds: "No medications extracted yet", sExam: "No examinations extracted yet",
    sAllergies: "No allergies extracted yet", sPmh: "No past medical history extracted",
    sPlan: "No plan extracted yet",
    sNotes: "Nothing flagged",
  };
  Object.entries(defaults).forEach(([id, t]) => fill(id, [], t));
}

// ---------- Copy / save ----------
function copyRecord() {
  const r = state.record;
  if (!r) return showError("Nothing to copy yet. Process a transcript first.");
  const pd = r.patient_details || {};
  const list = (x) => arr(x).map((i) => "  - " + (typeof i === "string" ? i : fmtDose(i))).join("\n");
  const lines = [
    `Patient: ${pd.name || "-"} | Age: ${pd.age || "-"} | Sex: ${pd.sex || "-"}`,
    `Chief complaint: ${r.chief_complaint || "-"}`,
    `Symptoms:\n${list(r.symptoms?.positive)}`,
    `Denies:\n${list(r.symptoms?.negative)}`,
    `Allergies:\n${list(r.medication_history?.allergies)}`,
    `Assessment: ${r.assessment?.provisional_diagnosis || "-"}`,
    `Investigations:\n${list(r.plan?.investigations)}`,
    `Rx:\n${list(r.plan?.prescriptions)}`,
    `Advice:\n${list(r.plan?.advice)}`,
    `Follow-up: ${r.plan?.follow_up || "-"}`,
  ];
  navigator.clipboard.writeText(lines.join("\n"));
  $("btnCopy").textContent = "Copied ✓";
  setTimeout(() => ($("btnCopy").textContent = "Copy to Prescription"), 1500);
}

function saveRecording() {
  if (!state.segments.length) return showError("No audio recorded yet.");
  const total = state.segments.reduce((n, s) => n + s.length, 0);
  const pcm = new Float32Array(total);
  let off = 0;
  state.segments.forEach((s) => { pcm.set(s, off); off += s.length; });
  const blob = encodeWav(pcm, 16000);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `consultation-${Date.now()}.wav`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, samples.length * 2, true);
  samples.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
  return new Blob([buf], { type: "audio/wav" });
}

// ---------- helpers ----------
function setStatus(t) { $("status").textContent = t; }
function showError(t) { const e = $("error"); e.textContent = t; e.hidden = false; }
function hideError() { $("error").hidden = true; }

$("btnStart").onclick = () => startMic().catch((e) => { showError("Mic/VAD error: " + e.message); setStatus(""); });
$("btnStop").onclick = stopMic;
$("btnClear").onclick = clearTranscript;
$("btnSave").onclick = saveRecording;
$("btnProcess").onclick = processTranscript;
$("btnClearSummary").onclick = clearSummary;
$("btnCopy").onclick = copyRecord;