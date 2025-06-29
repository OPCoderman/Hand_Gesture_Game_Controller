async function getPredictedLabel(landmarks) {
  try {
    const formatted = landmarks.map(lm => [lm.x, lm.y]);

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ landmarks: formatted })
    });

    if (!response.ok) {
      console.error("API error:", response.statusText);
      return null;
    }

    const result = await response.json();
    console.log("Predicted hand sign:", result.hand_sign);
    return result.hand_sign || null;
  } catch (error) {
    console.error("Prediction error:", error);
    return null;
  }
}
