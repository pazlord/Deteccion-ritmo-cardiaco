import cv2
import numpy as np
from scipy import signal
from collections import deque
import time

# ---------------- Parameters ----------------
BUFFER_SIZE = 150
ALPHA = 30.0
LOW_CUT = 0.8
HIGH_CUT = 3.0
AMPLIFY_CHROMA_ONLY = True
ROI_SIZE = (120, 120)  # fixed ROI size for buffer consistency

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Cannot open webcam")

fps = cam.get(cv2.CAP_PROP_FPS)
if fps <= 1 or np.isnan(fps):
    fps = 30.0  # assume 30 if low or unknown
print(f"Detected FPS: {fps:.1f}")

frame_buffer = deque(maxlen=BUFFER_SIZE)
green_signal = deque(maxlen=BUFFER_SIZE)

# Safe bandpass creator
def make_bandpass(fs):
    nyq = fs / 2.0
    low = LOW_CUT / nyq
    high = HIGH_CUT / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    if high <= low:
        high = min(low * 1.5, 0.99)
        low = max(high / 3.0, 0.001)
        print(f"[Warning] FPS too low ({fs:.2f}). Adjusted band: {low*nyq:.2f}-{high*nyq:.2f} Hz")
    numtaps = min(129, BUFFER_SIZE) | 1
    return signal.firwin(numtaps, [low, high], pass_zero=False)

bandpass = make_bandpass(fps)

def magnify_color(frames_np):
    T = frames_np.shape[0]
    if T < len(bandpass):
        return frames_np[-1]
    filtered = np.zeros_like(frames_np)
    for c in range(3):
        filtered[..., c] = signal.lfilter(bandpass, [1], frames_np[..., c], axis=0)
    amplified = frames_np + ALPHA * filtered
    return np.clip(amplified, 0.0, 1.0)[-1]

def estimate_bpm(signal_data, fs):
    if len(signal_data) < BUFFER_SIZE:
        return None
    sig = np.array(signal_data) - np.mean(signal_data)
    freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    mask = (freqs >= LOW_CUT) & (freqs <= HIGH_CUT)
    if np.any(mask):
        dominant = freqs[mask][np.argmax(fft_mag[mask])]
        bpm = dominant * 60.0
        return bpm
    return None

fps_counter = []
while True:
    ret, frame = cam.read()
    if not ret:
        break

    now = time.time()
    fps_counter.append(now)
    fps_counter = [t for t in fps_counter if now - t <= 2.0]
    current_fps = len(fps_counter) / 2.0
    fs = max(10.0, current_fps)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    bpm_text = "Detecting..."

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, ROI_SIZE)  # ensures consistent shape

        roi_float = roi_color.astype(np.float32) / 255.0
        frame_buffer.append(roi_float.copy())
        green_signal.append(np.mean(roi_float[..., 1]))

        if len(frame_buffer) == BUFFER_SIZE:
            frames_np = np.stack(frame_buffer, axis=0)
            if AMPLIFY_CHROMA_ONLY:
                y_channel = np.mean(frames_np, axis=3, keepdims=True)
                chroma = frames_np - y_channel
                magnified = y_channel + magnify_color(chroma)
            else:
                magnified = magnify_color(frames_np)

            mag_frame = (magnified * 255).astype(np.uint8)
            mag_frame_resized = cv2.resize(mag_frame, (w, h))
            frame[y:y+h, x:x+w] = mag_frame_resized

            bpm = estimate_bpm(green_signal, fs)
            if bpm:
                bpm_text = f"{bpm:.1f} BPM"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    else:
        bpm_text = "No face detected"

    cv2.putText(frame, bpm_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imshow("Real-time Heart Rate", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
