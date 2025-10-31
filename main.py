"""
Script para la detección de la frecuencia cardíaca (BPM) en tiempo real 
utilizando una cámara web, OpenCV y análisis de señal (rPPG).

El script detecta un rostro, aísla la región de la frente (ROI), 
extrae la señal promedio del canal verde (sensible a los cambios de 
flujo sanguíneo), la filtra y aplica una Transformada Rápida de Fourier (FFT) 
para encontrar la frecuencia dominante, que se convierte a BPM.
"""

# 1. Importaciones de bibliotecas
# Importaciones de terceros (comúnmente agrupadas primero)
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq

# Importaciones de la biblioteca estándar
import time
from collections import deque

# 2. Configuraciones y Constantes Globales
# PEP 8 sugiere usar MAYUSCULAS_CON_GUION_BAJO para constantes.

# Duración del búfer de señal en segundos
BUFFER_SECONDS = 10  
# Rango de BPM esperado (mínimo y máximo)
MIN_BPM = 45.0
MAX_BPM = 200.0
# Ruta al archivo Haar Cascade (obsoleto, se usa la ruta de cv2.data)
# FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
# Factor de ajuste de saturación
SAT_ADJUST = 2

# 3. Inicialización
# Cargar el clasificador Haar Cascade para la detección de rostros
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    exit()

# Validación e inicialización de la cámara
cam = cv2.VideoCapture(0)  # 0 para la cámara web predeterminada
if not cam.isOpened():
    exit()

# Intentar obtener FPS reales de la cámara, con un valor predeterminado
try:
    fps_cam = cam.get(cv2.CAP_PROP_FPS)
    if fps_cam < 1:
        fps_cam = 30.0  # Usar 30 si la cámara devuelve un valor inválido
except:
    fps_cam = 30.0  # Valor predeterminado si get() falla
    
# Calcular el tamaño del búfer basado en los FPS y la duración deseada
BUFFER_SIZE = int(BUFFER_SECONDS * fps_cam)

# Inicializar búferes (deques) para la señal y los timestamps
# Usar maxlen asegura que el búfer se desplace automáticamente
signal_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

# Variables de estado
bpm = 0.0
last_time = time.time()
face_detected = False

# 4. Bucle Principal de Procesamiento
while True:
    # Leer un frame de la cámara
    ret, frame = cam.read()
    if not ret:
        break
    
    # Voltear el frame horizontalmente (efecto espejo)
    frame = cv2.flip(frame, 1)
    now = time.time()

    # --- Aumento de Saturación ---
    # Convertir a HSV para manipular la saturación
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Aumentar la saturación (S) para mejorar la detección de color
    # Se usa float para evitar desbordamiento antes del recorte (clipping)
    s_float = s.astype(np.float32)
    s_float *= SAT_ADJUST
    s_float = np.clip(s_float, 0, 255)  # Limitar los valores al rango [0, 255]
    s = s_float.astype(np.uint8)
    
    # Volver a unir los canales HSV y convertir de nuevo a BGR
    hsv_modificado = cv2.merge([h, s, v])
    frame_saturado = cv2.cvtColor(hsv_modificado, cv2.COLOR_HSV2BGR)
    
    # Usar el frame saturado para la detección y visualización
    frame = frame_saturado

    # --- Detección de Rostro ---
    # Convertir a escala de grises para el clasificador Haar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        # Se encontró al menos un rostro
        face_detected = True
        
        # Tomar solo el primer rostro detectado
        (x, y, w, h) = faces[0]  
        
        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # --- Definición de la Región de Interés (ROI) - Frente ---
        # Se define la ROI como una sección en la parte superior del rostro
        roi_x1 = x + int(w * 0.30)
        roi_y1 = y + int(h * 0.10)
        roi_x2 = x + int(w * 0.70)
        roi_y2 = y + int(h * 0.25)

        # Asegurarse de que las coordenadas de la ROI estén dentro de los límites del frame
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(frame.shape[1], roi_x2)  # frame.shape[1] es el ancho
        roi_y2 = min(frame.shape[0], roi_y2)  # frame.shape[0] es el alto

        # Extraer la ROI del frame
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Dibujar el rectángulo de la ROI
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        
        # --- Extracción de la Señal (Promedio del Canal Verde) ---
        if roi_frame.size > 0:  
            # Separar los canales B, G, R de la ROI
            _, g, _ = cv2.split(roi_frame)
            # Calcular el promedio del canal verde
            green_mean = np.mean(g)
            
            # Añadir el valor y el timestamp a los búferes
            signal_buffer.append(green_mean)
            timestamps.append(now)
            
        else:  
            # Si la ROI es inválida (tamaño 0)
            face_detected = False
            
    else:  
        # No se detectó ningún rostro
        face_detected = False
        # Si se pierde el rostro, limpiar los búferes para reiniciar el cálculo
        signal_buffer.clear()
        timestamps.clear()

    # --- Procesamiento de la Señal y Cálculo de BPM ---
    # Procesar solo si hay un rostro y el búfer está lleno
    if face_detected and len(signal_buffer) == BUFFER_SIZE:
        
        # Calcular la frecuencia de muestreo (Fs) real
        total_time = timestamps[-1] - timestamps[0]
        fs = len(signal_buffer) / total_time
        
        # Convertir el búfer a un array numpy para el procesamiento
        signal = np.array(list(signal_buffer))
        
        # 1. Eliminar tendencia (Detrending): Quita cambios lentos (ej. luz)
        detrended_signal = detrend(signal)
        
        # 2. Filtrado (Bandpass): Aislar frecuencias de interés
        low_hz = MIN_BPM / 60.0
        high_hz = MAX_BPM / 60.0
        nyquist = 0.5 * fs  # Frecuencia de Nyquist
        
        # Obtener coeficientes del filtro Butterworth (orden 2)
        b, a = butter(2, [low_hz / nyquist, high_hz / nyquist], btype='band')
        # Aplicar el filtro (filtfilt no introduce desfase)
        filtered_signal = filtfilt(b, a, detrended_signal)
        
        # 3. Análisis de Frecuencia (FFT)
        N = len(filtered_signal)
        # Aplicar una ventana (Hanning) para reducir el 'leakage' espectral
        fft_window = filtered_signal * np.hanning(N)
        # Calcular la FFT real (rfft) y su magnitud
        fft_vals = np.abs(rfft(fft_window))
        # Obtener las frecuencias correspondientes
        fft_freqs = rfftfreq(N, 1.0 / fs)
        
        # 4. Encontrar la Frecuencia Pico (Dominante)
        # Filtrar el espectro de FFT para quedarnos solo con el rango de BPM válido
        valid_indices = np.where((fft_freqs >= low_hz) & (fft_freqs <= high_hz))
        valid_fft_vals = fft_vals[valid_indices]
        valid_fft_freqs = fft_freqs[valid_indices]
        
        if len(valid_fft_freqs) > 0:
            # Encontrar el índice de la magnitud más alta
            peak_index = np.argmax(valid_fft_vals)
            # Obtener la frecuencia correspondiente a ese pico
            peak_freq = valid_fft_freqs[peak_index]
            # Convertir la frecuencia (Hz) a BPM (Beats Per Minute)
            bpm = peak_freq * 60.0
        else:
            bpm = 0.0  # No se encontró pico en el rango válido
            
        # Imprimir el BPM en la consola (con retorno de carro para sobreescribir)
        print(f"BPM calculado: {bpm:.2f}      ", end='\r')
        
        # Limpiar los búferes para el siguiente cálculo completo
        signal_buffer.clear()
        timestamps.clear()
        
    
    # --- Lógica de Visualización (UI/Feedback) ---
    text_color = (0, 0, 255)  # Rojo por defecto
    
    if not face_detected:
        text = "No hay rostro detectado"
        # Limpiar búferes si se acaba de perder el rostro
        signal_buffer.clear()  
        timestamps.clear()
    elif len(signal_buffer) < BUFFER_SIZE:
        # Mostrar progreso de llenado del búfer
        percent_full = int(100 * len(signal_buffer) / BUFFER_SIZE)
        text = f"Calculando... {percent_full}%"
    else:
        # Mostrar el BPM calculado
        text = f"BPM: {bpm:.2f}"
        text_color = (0, 255, 0)  # Verde (éxito)
        
    # Dibujar el texto de estado/BPM en el frame
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    
    # Mostrar el frame resultante
    cv2.imshow('Camara', frame)

    # Condición de salida: presionar 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# 5. Limpieza Final
cam.release()
cv2.destroyAllWindows()