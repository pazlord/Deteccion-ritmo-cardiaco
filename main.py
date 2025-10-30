import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq

## Configurations 
BUFFER_SECONDS = 10
MIN_BPM = 45.0
MAX_BPM = 200.0
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    exit()

#Validation for camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    exit()

try:
    fps_cam = cam.get(cv2.CAP_PROP_FPS)
    if fps_cam < 1:
        fps_cam = 30
except:
    fps_cam = 30
    
BUFFER_SIZE = int(BUFFER_SECONDS * fps_cam)

signal_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

bpm = 0.0
last_time = time.time()
face_detected = False

satadj = 2

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    #Increase saturation
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_float = s.astype(np.float32)
    s_float *= satadj
    s_float = np.clip(s_float, 0, 255)
    s = s_float.astype(np.uint8)
    hsv_modificado = cv2.merge([h, s, v])
    frame_saturado = cv2.cvtColor(hsv_modificado, cv2.COLOR_HSV2BGR)

    now = time.time()
    
    #Detection face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame_saturado
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0] 
        face_detected = True

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_x1 = x + int(w * 0.30)
        roi_y1 = y + int(h * 0.10)
        roi_x2 = x + int(w * 0.70)
        roi_y2 = y + int(h * 0.25) 

        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(frame.shape[1], roi_x2)
        roi_y2 = min(frame.shape[0], roi_y2)

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2) 
        
        if roi_frame.size > 0: 
            _, g, _ = cv2.split(roi_frame)
            green_mean = np.mean(g)
            
            signal_buffer.append(green_mean)
            timestamps.append(now)
            
        else: 
            face_detected = False
            
    else: 
        face_detected = False
        signal_buffer.clear()
        timestamps.clear()

    # Process signal when buffer is full
    if face_detected and len(signal_buffer) == BUFFER_SIZE:
        
        total_time = timestamps[-1] - timestamps[0]
        fs = len(signal_buffer) / total_time #frequency 
        
        signal = np.array(list(signal_buffer))
        
        detrended_signal = detrend(signal)
        
        low_hz = MIN_BPM / 60.0
        high_hz = MAX_BPM / 60.0
        nyquist = 0.5 * fs
        
        # Get filter coefficients, and filter
        b, a = butter(2, [low_hz / nyquist, high_hz / nyquist], btype='band')
        filtered_signal = filtfilt(b, a, detrended_signal)
        
        # Get frequencies
        N = len(filtered_signal)
        fft_vals = np.abs(rfft(filtered_signal * np.hanning(N)))
        fft_freqs = rfftfreq(N, 1.0 / fs)
        
        # Find peak freq, and infer BPM
        valid_indices = np.where((fft_freqs >= low_hz) & (fft_freqs <= high_hz))
        valid_fft_vals = fft_vals[valid_indices]
        valid_fft_freqs = fft_freqs[valid_indices]
        
        if len(valid_fft_freqs) > 0:
            peak_index = np.argmax(valid_fft_vals)
            peak_freq = valid_fft_freqs[peak_index]
            bpm = peak_freq * 60.0
        else:
            bpm = 0.0 
        print(f"BPM calculado: {bpm:.2f}         ", end='\r')  
        signal_buffer.clear()
        timestamps.clear()
        
    
    text_color = (0, 0, 255) 
    if not face_detected:
        text = "No hay rostro detectado"
        signal_buffer.clear() 
        timestamps.clear()
    elif len(signal_buffer) < BUFFER_SIZE:
        text = f"Calculando... {int(100 * len(signal_buffer) / BUFFER_SIZE)}%"
    else:
        text = f"BPM: {bpm:.2f}"
        text_color = (0, 255, 0) 
        
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    
    cv2.imshow('Camara', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()