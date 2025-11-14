const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const transcribeRecordingBtn = document.getElementById("transcribeRecordingBtn");
const diarizeRecordingBtn = document.getElementById("diarizeRecordingBtn");
const uploadDiarizeBtn = document.getElementById("uploadDiarizeBtn");
const transcriptBtn = document.getElementById("transcriptBtn");
const downloadBtn = document.getElementById("downloadBtn");

const fileInput = document.getElementById("fileInput");
const audioPlayer = document.getElementById("audioPlayer");
const transcriptBox = document.getElementById("transcriptBox");
const diarizedBox = document.getElementById("diarizedBox");
const statusText = document.getElementById("statusText");
const timerText = document.getElementById("timerText");

let mediaRecorder;
let audioChunks = [];
let recordedBlob;
let uploadedFilePath = "";
let timerInterval;
let secondsElapsed = 0;

function updateStatus(message) {
  statusText.textContent = `Status: ${message}`;
}

function startTimer() {
  secondsElapsed = 0;
  timerText.textContent = "00:00";
  timerInterval = setInterval(() => {
    secondsElapsed++;
    const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
    const secs = String(secondsElapsed % 60).padStart(2, "0");
    timerText.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
}

// Start recording
startBtn.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(audioChunks, { type: "audio/wav" });
    audioPlayer.src = URL.createObjectURL(recordedBlob);
    updateStatus("Recording stopped");
    stopTimer();
  };

  mediaRecorder.start();
  updateStatus("Recording...");
  startTimer();
});

// Stop recording
stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    stopTimer();
    updateStatus("Recording stopped");
  }
});

// Transcribe recorded audio
transcribeRecordingBtn.addEventListener("click", async () => {
  if (!recordedBlob) return alert("No recording available.");

  const formData = new FormData();
  formData.append("file", recordedBlob, "recording.wav");

  try {
    updateStatus("Uploading...");
    const uploadRes = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData,
    });
    const uploadData = await uploadRes.json();
    uploadedFilePath = uploadData.path;

    updateStatus("Transcribing...");
    const transcribeRes = await fetch("http://127.0.0.1:5000/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: uploadedFilePath }),
    });
    const transcribeData = await transcribeRes.json();
    transcriptBox.value = transcribeData.transcript;
    updateStatus("Done!");
  } catch (err) {
    console.error(err);
    updateStatus("Error occurred");
    alert("Error during transcription.");
  }
});

// Diarize recorded audio
diarizeRecordingBtn.addEventListener("click", async () => {
  if (!recordedBlob) return alert("No recording available.");

  const formData = new FormData();
  formData.append("file", recordedBlob, "recording.wav");

  try {
    updateStatus("Uploading...");
    const uploadRes = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData,
    });
    const uploadData = await uploadRes.json();
    uploadedFilePath = uploadData.path;

    updateStatus("Diarizing...");
    const diarizeRes = await fetch("http://127.0.0.1:5000/diarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: uploadedFilePath }),
    });
    const diarizeData = await diarizeRes.json();
    diarizedBox.value = diarizeData.diarized.join("\n");
    updateStatus("Done!");
  } catch (err) {
    console.error(err);
    updateStatus("Error occurred");
    alert("Error during diarization.");
  }
});

// Upload and diarize selected file
uploadDiarizeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return alert("Please select an audio file.");

  const formData = new FormData();
  formData.append("file", file);

  try {
    updateStatus("Uploading...");
    const uploadRes = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData,
    });
    const uploadData = await uploadRes.json();
    uploadedFilePath = uploadData.path;

    updateStatus("Diarizing...");
    const diarizeRes = await fetch("http://127.0.0.1:5000/diarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: uploadedFilePath }),
    });
    const diarizeData = await diarizeRes.json();
    diarizedBox.value = diarizeData.diarized.join("\n");
    updateStatus("Done!");
  } catch (err) {
    console.error(err);
    updateStatus("Error occurred");
    alert("Error during upload or diarization.");
  }
});

// Transcribe selected file
transcriptBtn.addEventListener("click", async () => {
  if (!uploadedFilePath) return alert("Upload a file first.");

  try {
    updateStatus("Transcribing...");
    const transcribeRes = await fetch("http://127.0.0.1:5000/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: uploadedFilePath }),
    });
    const transcribeData = await transcribeRes.json();
    transcriptBox.value = transcribeData.transcript;
    updateStatus("Done!");
  } catch (err) {
    console.error(err);
    updateStatus("Error occurred");
    alert("Error during transcription.");
  }
});

// Download recorded or uploaded audio
