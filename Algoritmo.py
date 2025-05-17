import cv2
import time
import RPi.GPIO as GPIO
import numpy as np
from ultralytics import YOLO

# Configuración de pines GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Definición de pines
BUZZER_PIN = 17
LED_PINS = list(range(2, 22))  # 20 LEDs conectados a los pines GPIO 2-21
PIR_SENSOR_PIN = 23

# Inicialización de los pines GPIO
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(PIR_SENSOR_PIN, GPIO.IN)
for led_pin in LED_PINS:
    GPIO.setup(led_pin, GPIO.OUT)

# Función para calcular la distancia aproximada basada en el tamaño del objeto
def calculate_distance(box_width, box_height, frame_width, frame_height):
    # Valores de calibración (deberían ajustarse según el lente de la cámara y el tamaño promedio del felino)
    KNOWN_WIDTH = 0.6  # Ancho promedio de un felino en metros
    FOCAL_LENGTH = 800  # Valor a calibrar para tu cámara específica
    
    # Calculamos la distancia basada en el ancho relativo del objeto detectado
    object_width_ratio = box_width / frame_width
    estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / (object_width_ratio * frame_width)
    
    return estimated_distance

# Función para activar la alarma y los LEDs
def activate_alarm():
    print("¡ALARMA ACTIVADA!")
    # Encender todos los LEDs
    for led_pin in LED_PINS:
        GPIO.output(led_pin, GPIO.HIGH)
    
    # Activar el buzzer (patrón de alarma)
    for _ in range(5):  # Repetir 5 veces
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.2)

# Función para desactivar la alarma y los LEDs
def deactivate_alarm():
    # Apagar todos los LEDs
    for led_pin in LED_PINS:
        GPIO.output(led_pin, GPIO.LOW)
    
    # Asegurar que el buzzer está apagado
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Cargar el modelo YOLO
model = YOLO('./PruebaModelo/model/best_felinos_50.pt')

alarm_active = False
last_detection_time = 0
detection_cooldown = 30  # segundos entre activaciones de alarma

try:
    while True:
        # Verificar si el sensor PIR detecta movimiento
        motion_detected = GPIO.input(PIR_SENSOR_PIN)
        
        if not motion_detected:
            time.sleep(0.1)  # Pequeña pausa para no sobrecargar el CPU
            continue
        
        # Si detectamos movimiento, capturamos un frame
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame")
            break
            
        frame_height, frame_width = frame.shape[:2]
        
        # Ejecutar detección con YOLO
        results = model.predict(frame, imgsz=640, conf=0.4)  # Aumentamos la confianza
        
        felino_detected = False
        felino_distance = float('inf')
        
        if len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Obtener coordenadas de la caja
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    # Calcular distancia aproximada
                    distance = calculate_distance(box_width, box_height, frame_width, frame_height)
                    felino_distance = min(felino_distance, distance)
                    
                    # Mostrar distancia en la imagen
                    cv2.putText(frame, f"Dist: {distance:.1f}m", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    felino_detected = True
            
            # Anotar el frame con las detecciones
            annotated_frame = results[0].plot()
            
            current_time = time.time()
            
            # Verificar condiciones para activar la alarma
            if (felino_detected and 
                felino_distance <= 20.0 and  # Felino a menos de 20 metros
                motion_detected and          # Sensor de movimiento activado
                (current_time - last_detection_time) > detection_cooldown):
                
                if not alarm_active:
                    activate_alarm()
                    alarm_active = True
                    last_detection_time = current_time
                    
                    # Guardar la imagen de la detección
                    cv2.imwrite(f"deteccion_felino_{int(current_time)}.jpg", annotated_frame)
            else:
                if alarm_active and (current_time - last_detection_time) > 10:  # Desactivar después de 10 segundos
                    deactivate_alarm()
                    alarm_active = False
                
            # Mostrar información en la imagen
            status_text = "ALARMA ACTIVA" if alarm_active else "Monitoreando"
            cv2.putText(annotated_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if alarm_active else (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"Distancia: {felino_distance:.1f}m", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Mostrar la imagen con anotaciones
            cv2.imshow('Detección de Felinos', annotated_frame)
        else:
            cv2.imshow('Detección de Felinos', frame)
            
            # Si no hay detección por un tiempo, desactivar alarma
            if alarm_active and (time.time() - last_detection_time) > 10:
                deactivate_alarm()
                alarm_active = False
        
        # Salir con la tecla ESC
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

except KeyboardInterrupt:
    print("Programa detenido por el usuario")
finally:
    # Limpieza al salir
    cap.release()
    cv2.destroyAllWindows()
    deactivate_alarm()
    GPIO.cleanup()