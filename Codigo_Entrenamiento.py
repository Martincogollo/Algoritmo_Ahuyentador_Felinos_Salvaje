from ultralytics import YOLO
import os

def main():
    # ruta base del proyecto
    project_base = os.path.abspath(os.path.dirname(__file__))
    print(f"Directorio base del proyecto: {project_base}")

    # Se carga el modelo
    model = YOLO('./ModeloPrueba/model/yolov8n.pt') 

    # Ruta del archivo data.yaml
    data_yaml_path = os.path.join(project_base,'datatrain', 'data.yaml')

    # Verifico que el archivo exista
    if os.path.exists(data_yaml_path):
        print(f"Usando archivo de configuración: {data_yaml_path}")
        
        # Imprimir contenido del archivo para verificar
        with open(data_yaml_path, 'r') as f:
            print("Contenido del archivo data.yaml:")
            print(f.read())
        
        # Entrenar el modelo
        try:
            results = model.train(
                data=data_yaml_path,  # Archivo de configuración de datos
                epochs=60,           # Número de épocas
                imgsz=640,            # Tamaño de la imagen
                device=0,
                batch=4,              # Tamaño del lote
                patience=5,          # Parametro para detener si no mejora después de 5 épocas
                verbose=True       
            )
            
            print("Entrenamiento completado.")
            print(f"Resultados guardados en: {model.ckpt_path}")
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
    else:
        print(f"ERROR: El archivo no existe en la ruta: {data_yaml_path}")

if __name__ == '__main__':
    main()
