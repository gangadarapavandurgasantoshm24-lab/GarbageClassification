const dropzone = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const emptyPreview = document.getElementById("emptyPreview");
const classifyBtn = document.getElementById("classifyBtn");
const clearBtn = document.getElementById("clearBtn");
const errorBox = document.getElementById("errorBox");
const placeholder = document.getElementById("placeholder");
const resultContent = document.getElementById("resultContent");
const predictedClass = document.getElementById("predictedClass");
const confidenceScore = document.getElementById("confidenceScore");
const warningText = document.getElementById("warningText");
const probabilityList = document.getElementById("probabilityList");
const originalOutput = document.getElementById("originalOutput");
const gradcamOutput = document.getElementById("gradcamOutput");
const latency = document.getElementById("latency");

let selectedFile = null;

function setError(message) {
  errorBox.textContent = message || "";
}

function setLoading(isLoading) {
  classifyBtn.disabled = isLoading || !selectedFile;
  classifyBtn.classList.toggle("loading", isLoading);
  classifyBtn.querySelector(".btn-text").textContent = isLoading ? "Classifying..." : "Classify";
}

function resetResult() {
  placeholder.style.display = "grid";
  resultContent.style.display = "none";
  latency.textContent = "-- ms";
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  previewImage.style.display = "none";
  emptyPreview.style.display = "block";
  classifyBtn.disabled = true;
  setError("");
  resetResult();
}

function handleFile(file) {
  setError("");
  resetResult();
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setError("Please select a valid image file.");
    return;
  }
  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  previewImage.style.display = "block";
  emptyPreview.style.display = "none";
  classifyBtn.disabled = false;
}

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  handleFile(event.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
});

clearBtn.addEventListener("click", clearSelection);

function renderProbabilities(probabilities, finalLabel, adjustedPrediction) {
  probabilityList.innerHTML = "";
  const entries = Object.entries(probabilities).sort((a, b) => {
    if (adjustedPrediction) {
      if (a[0] === finalLabel) return -1;
      if (b[0] === finalLabel) return 1;
    }
    return b[1] - a[1];
  });
  entries.forEach(([label, probability]) => {
    const percent = Math.round(probability * 1000) / 10;
    const row = document.createElement("div");
    row.className = label === finalLabel ? "prob-row selected" : "prob-row";
    row.innerHTML = `
      <span>${label}</span>
      <div class="bar"><span style="width:${percent}%"></span></div>
      <b>${percent.toFixed(1)}%</b>
    `;
    probabilityList.appendChild(row);
  });
}

classifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setLoading(true);
  setError("");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    placeholder.style.display = "none";
    resultContent.style.display = "block";
    predictedClass.textContent = data.predicted_class;
    confidenceScore.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    if (data.low_confidence || data.adjusted_prediction) {
      warningText.textContent = data.message;
      warningText.style.display = "block";
    } else {
      warningText.textContent = "";
      warningText.style.display = "none";
    }
    latency.textContent = `${data.processing_time_ms} ms`;
    renderProbabilities(data.probabilities, data.predicted_class, data.adjusted_prediction);

    originalOutput.src = data.image_url;
    gradcamOutput.src = data.gradcam_url;
    originalOutput.style.display = "block";
    gradcamOutput.style.display = "block";
  } catch (error) {
    setError(error.message);
  } finally {
    setLoading(false);
  }
});
