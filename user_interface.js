// user_interface.js

document.addEventListener("DOMContentLoaded", () => {
  const spotSelect = document.getElementById("spot-id");
  const daySelect = document.getElementById("day");
  const timeInput = document.getElementById("time");
  const timeSlider = document.getElementById("myRange");
  const submitButton = document.getElementById("submit");
  
  // Display elements
  const selectedSpotDisplay = document.getElementById("selected-spot");
  const selectedDayDisplay = document.getElementById("selected-day");
  const selectedTimeDisplay = document.getElementById("selected-time");

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

  // Update display elements
  function updateDisplay() {
    selectedSpotDisplay.textContent = spotSelect.value === "null" ? "None" : spotSelect.value;
    selectedDayDisplay.textContent = daySelect.value === "null" ? "None" : daySelect.value;
    selectedTimeDisplay.textContent = timeInput.value;
  }

  // Initialize display
  updateDisplay();

  // Listen for messages from the map iframe
  window.addEventListener('message', function(event) {
    if (event.data.type === 'spotSelected') {
      const spotId = event.data.spotId;
      const spotNumber = event.data.spotNumber;
      
      // Update the spot select dropdown
      spotSelect.value = spotId;
      
      // Update display
      updateDisplay();
      
      // Show feedback
      showNotification(`Parking Spot ${spotNumber} selected!`);
    }
  });

  // Show notification function
  function showNotification(message) {
    // Remove any existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => {
          if (notification.parentNode) {
            document.body.removeChild(notification);
          }
        }, 300);
      }
    }, 3000);
  }

  // Synchronize slider → time input
  timeSlider.addEventListener("input", () => {
    const sliderValue = parseInt(timeSlider.value);
    const syncedTime = sliderToTime(sliderValue);
    timeInput.value = syncedTime;
    updateDisplay();
  });

  // Synchronize time input → slider
  timeInput.addEventListener("input", () => {
    const sliderValue = timeToSlider(timeInput.value);
    timeSlider.value = sliderValue;
    updateDisplay();
  });

  // Update display when selections change
  spotSelect.addEventListener("change", updateDisplay);
  daySelect.addEventListener("change", updateDisplay);

  // Submit button
  submitButton.addEventListener("click", () => {
    const selectedSpot = spotSelect.value;
    const selectedDay = daySelect.value;
    const selectedTime = timeInput.value; // this is always synced with slider

    // Validate inputs
    if (selectedSpot === "null" || selectedDay === "null") {
      showNotification("Please select both a spot and a day before submitting.");
      return;
    }

    const data = {
      spot: selectedSpot,
      day: selectedDay,
      time: selectedTime,
    };

    console.log("Submitted Data:", data);

    // Show success message
    submitButton.textContent = "Submitted!";
    submitButton.style.backgroundColor = "#fffed0";
    submitButton.style.color = "#14222b";
    
    setTimeout(() => {
      submitButton.textContent = "Submit";
      submitButton.style.backgroundColor = "#c7f361";
      submitButton.style.color = "#14222b";
    }, 2000);

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
