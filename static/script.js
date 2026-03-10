document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const formContainer = form.closest(".glass-panel");
  const resultContainer = document.getElementById("result-container");
  const resetBtn = document.getElementById("reset-btn");

  // Result elements
  const probabilityValue = document.getElementById("probability-value");
  const churnStatus = document.getElementById("churn-status");
  const resultIcon = document.getElementById("result-icon");
  const resultDesc = document.getElementById("result-description");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Disable button and show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = "Analyzing...";
    submitBtn.disabled = true;

    try {
      // Send request to API
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const result = await response.json();

      if (result.status === "success") {
        displayResult(result);

        // Transition UI
        formContainer.classList.add("hidden");
        resultContainer.classList.remove("hidden");
      } else {
        alert("Error during prediction: " + result.message);
      }
    } catch (error) {
      console.error("Error:", error);
      alert(
        "Failed to connect to the prediction server. Make sure it is running.",
      );
    } finally {
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", () => {
    resultContainer.classList.add("hidden");
    formContainer.classList.remove("hidden");
    // Reset the icon classes
    resultIcon.className = "result-icon";
  });

  function displayResult(result) {
    const isChurning = result.prediction === "Yes";

    // Update values
    probabilityValue.textContent = result.probability;

    // Parse probability string to float (e.g. "82.50%" -> 82.50)
    const probNum = parseFloat(result.probability.replace("%", ""));

    // Set dynamic styling based on prediction
    if (isChurning) {
      // High Churn Risk
      churnStatus.textContent = "High Churn Risk";
      churnStatus.className = "result-status status-danger";
      resultIcon.className = "result-icon danger";
      probabilityValue.style.color = "var(--danger)";

      // Icon: Alert / Warning / Trending Down
      resultIcon.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;

      resultDesc.textContent = `The model indicates this customer is highly likely to cancel their service. Action required to retain them.`;
    } else {
      // Low Churn Risk / Retained
      churnStatus.textContent = "Likely Retained";
      churnStatus.className = "result-status status-success";
      resultIcon.className = "result-icon success";
      probabilityValue.style.color = "var(--success)";

      // Icon: Check / Protect / Shield
      resultIcon.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>`;

      resultDesc.textContent = `The model suggests this customer is satisfied and is projected to remain subscribed.`;
    }

    // Count up animation for probability
    animateValue(probabilityValue, 0, probNum, 1000, "%");
  }

  // Number counting animation
  function animateValue(obj, start, end, duration, suffix) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // Ease out quad
      const easeProgress = progress * (2 - progress);
      obj.innerHTML =
        (start + easeProgress * (end - start)).toFixed(1) + suffix;
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        obj.innerHTML = end.toFixed(2) + suffix; // Guarantee final value is accurate
      }
    };
    window.requestAnimationFrame(step);
  }
});
