console.log("JS LOADED SUCCESSFULLY");
let selectedFile = null;
document.addEventListener("DOMContentLoaded", () => {

  const stars = document.querySelectorAll(".star");
  const feedbackText = document.getElementById("feedbackText");
  const submitBtn = document.getElementById("submitFeedback");
  const reviewsList = document.getElementById("reviewsList");

  if (stars.length) {
    let selectedRating = 0;

    stars.forEach(star => {
      star.addEventListener("click", () => {
        selectedRating = parseInt(star.getAttribute("data-rating"));
        stars.forEach(s => s.classList.remove("selected"));

        for (let s of stars) {
          s.classList.add("selected");
          if (s === star) break;
        }
      });
    });

    submitBtn.addEventListener("click", () => {
      const feedback = feedbackText.value.trim();

      if (!selectedRating) return alert("Please select rating");
      if (!feedback) return alert("Write feedback");

      const li = document.createElement("li");
      li.innerHTML = `<strong>★${selectedRating}</strong> — ${feedback}`;
      reviewsList.prepend(li);

      feedbackText.value = "";
      stars.forEach(s => s.classList.remove("selected"));
      selectedRating = 0;
    });
  }
});


// 🔥 MAIN DETECTION FUNCTION
async function runDetection() {

  if (!selectedFile) {
    alert("Please upload an image first!");
    return;
  }
  const file = selectedFile;

  if (!file) {
    alert("Upload image first!");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    console.log("Backend Response:", data); // 🔥 DEBUG

    showResults(data);

  } catch (err) {
    console.error(err);
    alert("Backend not running!");
  }
}


// 🔥 SHOW REAL RESULTS
function showResults(data) {

  const isFake = data.result === "Deepfake";

  const val   = document.getElementById('verdictValue');
  const conf  = document.getElementById('verdictConf');
  const score = document.getElementById('verdictScore');
  const icon  = document.getElementById('verdictIcon');
  const card  = document.getElementById('verdictCard');

  if (isFake) {
    val.textContent = '⚠ DEEPFAKE DETECTED';
    icon.innerHTML = '<i class="fas fa-times-circle"></i>';
    card.className = 'verdict-card fake';
  } else {
    val.textContent = '✓ AUTHENTIC';
    icon.innerHTML = '<i class="fas fa-check-circle"></i>';
    card.className = 'verdict-card authentic';
  }

  conf.textContent = `Confidence: ${data.confidence}%`;
  score.textContent = `${Math.round(data.confidence)}%`;
}