# ML Course Final Project - Gesture Control Game

This project implements a gesture-controlled game using machine learning for gesture recognition. Players can control the game using hand gestures captured through their webcam.

## 🎮 Controls

The game can be controlled through:
- Hand gestures (via webcam)
- Keyboard arrows (as fallback)
  
## 📊 Prometheus Monitoring Queries for Hand Sign Prediction API

This section lists all Prometheus queries used to monitor the performance, usage, and system metrics related to the hand sign prediction FastAPI application.
🚀 Application-Level Metrics
✅ 1. Filter Metrics by HTTP 200 Status

{http_status="200"}

Description: Filters all metrics where the HTTP status code is 200 (successful response).
📈 2. Rate of Successful /predict Requests (5-Min Window)

sum(rate(hand_sign_requests_total{endpoint="/predict", http_status=~"2.."}[5m]))

Description: Calculates the rate of successful (2xx) requests to the /predict endpoint over the last 5 minutes.
🔍 3. Filter Metrics for Specific Endpoint and Instance

{endpoint="/predict", instance="app:8000", job="fastapi_app"}

Description: Filters all metrics for the /predict endpoint running on the app:8000 instance of the fastapi_app job.
📉 4. Rate of All /predict Requests (1-Min Window)

sum(rate(hand_sign_requests_total{endpoint="/predict"}[1m]))

Description: Computes the total rate of all requests to the /predict endpoint over a 1-minute window, regardless of status code.
🖥️ System-Level Metrics (Windows Exporter)
💾 5. Used Physical Memory on Windows Host

windows_memory_physical_total_bytes{instance="192.168.68.55:9182"} - windows_memory_available_bytes{instance="192.168.68.55:9182"}

Description: Calculates the amount of used physical memory by subtracting available memory from the total on a Windows machine.

