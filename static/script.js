document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const formContainer = form.closest(".glass-panel");
  const resultContainer = document.getElementById("result-container");
  const resetBtn = document.getElementById("reset-btn");
  const submitBtn = document.getElementById("submit-btn");
  const spinner = document.getElementById("loading-spinner");

  // Result elements
  const probabilityValue = document.getElementById("probability-value");
  const churnStatus = document.getElementById("churn-status");
  const resultIcon = document.getElementById("result-icon");
  const resultDesc = document.getElementById("result-description");
  const confidenceFill = document.getElementById("confidence-fill");

  // ── Client-side validation ──────────────────────────────────────────────
  const NUMERIC_RANGES = {
    MonthlyMinutes:       [0, 10000],
    OverageMinutes:       [0, 5000],
    MonthlyRevenue:       [0, 5000],
    TotalRecurringCharge: [0, 5000],
    IncomeGroup:          [1, 9],
    AgeHH1:               [0, 120],
    MonthsInService:      [0, 600],
    CurrentEquipmentDays: [0, 3000],
    RoamingCalls:         [0, 1000],
  };

  function validateForm(data) {
    const errors = [];
    for (const [field, [min, max]] of Object.entries(NUMERIC_RANGES)) {
      const raw = data[field];
      if (raw === "" || raw === undefined || raw === null) continue;
      const val = parseFloat(raw);
      if (isNaN(val)) {
        errors.push(`${field} must be a number`);
      } else if (val < min || val > max) {
        errors.push(`${field} must be between ${min} and ${max} (entered: ${val})`);
      }
    }
    return errors;
  }

  // ── Form submit ─────────────────────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Client-side validation
    const validationErrors = validateForm(data);
    if (validationErrors.length > 0) {
      alert("Please fix the following:\n\n" + validationErrors.join("\n"));
      return;
    }

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.style.opacity = "0.5";
    spinner.classList.remove("hidden");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (!response.ok || result.status !== "success") {
        const msg = result.errors
          ? result.errors.join("\n")
          : result.message || "Prediction failed.";
        alert("Error: " + msg);
        return;
      }

      displayResult(result);
      formContainer.classList.add("hidden");
      resultContainer.classList.remove("hidden");

    } catch (error) {
      console.error("Error:", error);
      alert("Could not reach the prediction server. Make sure it is running.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.style.opacity = "";
      spinner.classList.add("hidden");
    }
  });

  // ── Reset ───────────────────────────────────────────────────────────────
  resetBtn.addEventListener("click", () => {
    resultContainer.classList.add("hidden");
    formContainer.classList.remove("hidden");
    resultIcon.className = "result-icon";
    confidenceFill.style.width = "0%";
  });

  // ── Display result ──────────────────────────────────────────────────────
  function displayResult(result) {
    const isChurning = result.prediction === "Yes";
    const probNum = parseFloat(result.probability.replace("%", ""));

    if (isChurning) {
      churnStatus.textContent = "High Churn Risk";
      churnStatus.className = "result-status status-danger";
      resultIcon.className = "result-icon danger";
      probabilityValue.style.color = "var(--danger)";
      confidenceFill.style.backgroundColor = "var(--danger)";
      resultIcon.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
      resultDesc.textContent = "The model indicates this customer is highly likely to cancel their service. Action required to retain them.";
    } else {
      churnStatus.textContent = "Likely Retained";
      churnStatus.className = "result-status status-success";
      resultIcon.className = "result-icon success";
      probabilityValue.style.color = "var(--success)";
      confidenceFill.style.backgroundColor = "var(--success)";
      resultIcon.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>`;
      resultDesc.textContent = "The model suggests this customer is satisfied and projected to remain subscribed.";
    }

    // Animate probability count-up
    animateValue(probabilityValue, 0, probNum, 1000, "%");

    // Animate confidence bar fill
    requestAnimationFrame(() => {
      confidenceFill.style.width = probNum + "%";
    });
  }

  // ── Count-up animation ──────────────────────────────────────────────────
  function animateValue(el, start, end, duration, suffix) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const ease = progress * (2 - progress); // ease-out quad
      el.innerHTML = (start + ease * (end - start)).toFixed(1) + suffix;
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.innerHTML = end.toFixed(2) + suffix;
      }
    };
    requestAnimationFrame(step);
  }
});
