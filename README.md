# recycling-image-classification

El objetivo de este proyecto es entrenar un modelo de visión por computadora que sea capaz de clasificar imágenes de residuos en seis categorías: biodegradable, glass, metal, non_recyclable, paper y plastic.

Este sistema busca resolver la incorrecta separación de desechos, debido a que esto reduce la eficiencia de los procesos de reciclaje, aumenta los costos operativos, genera contaminación cruzada y
evita la reusabilidad de materiales.

Al automatizar la identificación de residuos, se facilita la clasificación precisa y rápida de materiales, la asistencia a sistemas robóticos como NAO en tareas de selección
y la posibilidad de integrar la IA en flujos reales de gestión ambiental.

En conjunto, este proyecto busca mejorar las prácticas de reciclaje, promover una gestión sostenible de residuos y demostrar cómo el aprendizaje profundo puede aplicarse a problemas medioambientales reales.

## 1. Dataset
Se utilizó el Recyclable and Household Waste Classification Dataset, el cual contiene 15,000 imágenes organizadas en 30 categorías de residuos. Cada categoría incluye 500 imágenes, divididas equitativamente en dos tipos:
- default: fotos en condiciones controladas o de estudio.
- real_world: imágenes tomadas en escenarios reales, con variaciones de iluminación, contexto y ruido visual.
  
Link del dataset: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification?resource=download


## 2. Preparación de dataset
Para este entrenamiento, y debido al objetivo del proyecto, las imágenes se reorganizaron en las siguientes 6 categorías:
- biodegradable
- glass
- metal
- non_recyclable
- paper
- plastic

Para ello, se realizó lo siguiente:
- Recolección de imágenes desde las carpetas default/ y real_world/.
- Agrupación en las 6 clases finales utilizando CLASS_MAP.
- División en train (70%), validation (15%), test (15%) usando train_test_split.
- Reorganización física de todas las imágenes en las carpetas correspondientes.

## 3. Balance de clases
Se contabilizaron las imágenes por clase dentro del conjunto de prueba (test) y se calcularon los pesos de clase usando compute_class_weight.
El resultado mostró que el dataset está equilibrado, con pesos ≈ 1.0 para todas las clases, por lo que no fue necesario aplicar compensación adicional.

## 4. Data Augmentation
Para asegurar la generalización y evitar el overfitting para el training se utilizó data augmentation, que se encargó de agregar rotaciones, shifts, shear, zoom, flip horizontal 
y variaciones de brillo.

## 5. Arquitectura del Modelo
El modelo implementado es EfficientNetB0 como red base. Esta arquitectura ha sido ampliamente validada en tareas de clasificación de imágenes y proporciona un equilibrio óptimo entre rendimiento y eficiencia computacional.
- Se utilizaron pesos pre-entrenados en ImageNet, lo que permite aprovechar conocimientos previamente adquiridos sobre formas, texturas y estructuras visuales.
- Se estableció include_top=False para eliminar la capa de clasificación original del modelo.
- Durante la primera fase de entrenamiento, la red base se mantuvo congelada, evitando la actualización de sus pesos y permitiendo que únicamente se entrene la nueva cabeza del modelo.
Para la etapa de clasificación se añadieron las siguientes capas:
- GlobalAveragePooling2D: Se encarga de eliminar parámetros innecesarios y previene el sobreajuste.
- Dense (256 unidades, activación ReLU): Actúa como capa intermedia que aprende representaciones específicas del nuevo conjunto de 6 clases.
- Dense (6 unidades, activación Softmax): Capa final encargada de producir las probabilidades de pertenencia a cada una de las 6 clases de residuos.


## 6. Entrenamiento del modelo
Características principales:
- Optimizer: Adam (3e-4)
- Loss: Categorical Crossentropy
- Métrica: Accuracy
- Épocas: 20
- Callbacks: Se utiliza ModelCheckpoint que guarda el modelo durante entrenamiento.

## 7. Evaluación del modelo
Durante el entrenamiento, el modelo mostró una mejora progresiva del accuracy, iniciando el entrenamiento con un accuracy de 0.4932 y terminando con un accuracy de 0.9845.
Asimismos el train loss disminuyó llegando a valores inferiores a 0.06, lo que evidencia la capacidad del modelo para aprender del dataset. Sin embargo, al realizar la evaluación del modelo
el accuracy disminuyó a 0.87 y el loss aumentó a 0.39. 

Matriz de confusión:

```txt
[[162 1 1 1 7 5]
[ 1 134 5 0 2 11]
[ 4 8 159 1 1 4]
[ 2 0 2 154 11 8]
[ 6 0 5 14 189 5]
[ 6 16 11 5 9 189]]
```
- Las clases paper, metal y biodegradable muestran una cantidad alta de aciertos (189, 159 y 162 respectivamente) y pocos errores, indicando que el modelo ha aprendido patrones visuales claros para estas categorías.
- La clase plastic es la que presenta mayor confución, especialmente con glass (16), metal (11) y paper (9).
- Algunas clases tienden a confundirse entre sí debido a similitudes entre ellas. Como paper con non_recyclable (14 errores), glass con plastic (11 errores) y metal con glass (8 errores).


Métricas por clase:

```txt
                precision    recall  f1-score   support

 biodegradable       0.90      0.92      0.91       177
         glass       0.84      0.88      0.86       153
         metal       0.87      0.90      0.88       177
non_recyclable       0.88      0.87      0.88       177
         paper       0.86      0.86      0.86       219
       plastic       0.85      0.80      0.83       236

      accuracy                           0.87      1139
     macro avg       0.87      0.87      0.87      1139
  weighted avg       0.87      0.87      0.87      1139
```

  - Las clases biodegradable y metal presentan el f1-score y recall más altos, lo que indica que el modelo tiene alta capacidad para reconocer este tipo de residuo.
  - La clase plastic es muestra el rendimiento más bajo, esto se debe principalmente a que encuentra similitudes con categorías como glass y paper.
  - El desempeño general del modelo es consistente y mantiene un rendimiento equilibrado entre clases. Esto indica que el modelo generaliza de forma equilibrada entre todas las categorías.

## 8. Implementación en el NAO

Esta guía detalla los pasos para configurar y ejecutar un modelo de Inteligencia Artificial de clasificación de reciclaje usando un servidor FastAPI y el robot NAO v6 programado con Choreographe.

### Preparar y Ejecutar la API (Servidor de Inferencia)

El servidor de inferencia debe ejecutarse en un ordenador (PC o servidor) que esté en la misma red que el robot NAO.

A. Requisitos Previos e Instalación

Instalar Python 3.x y las librerías necesarias en tu ordenador: uvicorn, fastapi, tensorflow, numpy, opencv-python (cv2), y Pillow (PIL).

Comando de instalación:

pip install uvicorn fastapi tensorflow opencv-python Pillow


Archivos del Modelo: Coloca el archivo del modelo de IA, llamado best_model_2.h5, en la misma carpeta donde guardarás el script de la API.

B. Ejecutar el Servidor

Guarda el primer script (el de FastAPI) como un archivo Python (ejemplo: api_server.py).

Ejecútalo en tu terminal:

python api_server.py


Verificación: Debes ver mensajes de carga del modelo y la confirmación de Uvicorn, como Uvicorn running on http://0.0.0.0:8000.

IMPORTANTE: Este servidor DEBE permanecer activo y en ejecución (en primer plano o en segundo plano) durante todo el proceso.

### Configurar la Conexión en el Script del NAO (Cliente)

El script del NAO debe ser configurado para apuntar a la IP correcta de tu ordenador que ejecuta la API.

Identificar la IP: Averigua la dirección IP local de tu ordenador (servidor API). Por ejemplo: 192.168.1.50.

Ajustar el Script del NAO: Ajustar una linea dentro del script del NAO para que coincida con la IP de tu ordenador y el puerto de la API (8000):

Conexión del Robot: Asegúrate de que el robot NAO v6 esté conectado a la misma red (LAN o Wi-Fi) que el ordenador con la API.

### Implementación en Choreographe

Se utiliza el software Choreographe para cargar el script modificado y crear el comportamiento del robot.

Conectar el Robot: Abre Choreographe, ve a "Conexión" y conéctate a la IP del robot NAO.

Crear Proyecto: Crea un nuevo proyecto de comportamiento.

Añadir el Nodo de Script: Arrastra la caja "Python Script" (Nodo de Script) desde la librería al diagrama.

Insertar el Script:

Haz doble clic en el nodo "Python Script".

Borra el código predeterminado.

Pega el script completo del NAO, ya modificado con la IP correcta, en el editor.

Conectar la Entrada: Conecta la señal de inicio (ej. desde una caja On Start) a la entrada onStart del nodo de script para iniciar la ejecución.

### Ejecución y Verificación Final

Transferir y Ejecutar: En Choreographe, haz clic en el botón de "Reproducir" (Play) o "Transferir y Reproducir" (Transfer and Play).

Verificación en el Robot: El NAO comenzará el proceso:

Dirá: "Observando entorno."

Capturará la imagen de su cámara inferior (cameraID = 1).

Dirá: "Procesando imagen."

Verificación en el Servidor (API):

Revisa la consola de tu ordenador. Debes ver el mensaje: "FOTO GUARDADA: Revisa el archivo 'ultima_vision_nao.jpg' en tu carpeta."

La API mostrará la salida de la predicción de TensorFlow y enviará el resultado JSON al NAO.

Resultado Final: El NAO dirá el objeto detectado y dónde debe reciclarse (ejemplo: "He detectado plástico. Esto va en el contenedor amarillo."), completando el ciclo de identificación.

Video de pruebas con el NAO: https://www.youtube.com/watch?v=rL4mmIgELGg
