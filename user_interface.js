// user_interface.js

document.addEventListener("DOMContentLoaded", () => {
  const spotSelect = document.getElementById("spot-id");
  const daySelect = document.getElementById("day");
  const timeInput = document.getElementById("time");
  const timeSlider = document.getElementById("myRange");
  const submitButton = document.getElementById("submit");

  const sliderMin = 7; // 7 AM
  const sliderMax = 19; // 7 PM

  // Helper: Convert slider (1–100) → time string
  function sliderToTime(value) {
    const hours = sliderMin + (value / 100) * (sliderMax - sliderMin);
    const hourStr = Math.floor(hours).toString().padStart(2, "0");
    const minStr = Math.round((hours % 1) * 60)
      .toString()
      .padStart(2, "0");
    return `${hourStr}:${minStr}`;
  }

  // Helper: Convert time string (HH:MM) → slider value (1–100)
  function timeToSlider(timeStr) {
    const [h, m] = timeStr.split(":").map(Number);
    const decimalHours = h + m / 60;
    const sliderVal =
      ((decimalHours - sliderMin) / (sliderMax - sliderMin)) * 100;
    return Math.min(100, Math.max(1, Math.round(sliderVal)));
  }

  // Synchronize slider → time input
  timeSlider.addEventListener("input", () => {
    const sliderValue = parseInt(timeSlider.value);
    const syncedTime = sliderToTime(sliderValue);
    timeInput.value = syncedTime;
  });

  // Synchronize time input → slider
  timeInput.addEventListener("input", () => {
    const sliderValue = timeToSlider(timeInput.value);
    timeSlider.value = sliderValue;
  });

  // Submit button
  submitButton.addEventListener("click", () => {
    const selectedSpot = spotSelect.value;
    const selectedDay = daySelect.value;
    const selectedTime = timeInput.value; // this is always synced with slider

    const data = {
      spot: selectedSpot,
      day: selectedDay,
      time: selectedTime,
    };

    console.log("Submitted Data:", data);

    // Example: send to backend (optional)
    // fetch('/submit', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(data)
    // }).then(response => {
    //   if (response.ok) {
    //     console.log('Data submitted successfully');
    //   }
    // });
  });
});
