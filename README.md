# recycling-image-classification
Proyecto de clasificación de imágenes de desechos de reciclaje utilizando EfficientNet


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
- Agrupación en las 6 clases finales según CLASS_MAP.
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

- Matriz de confusión:
Los resultados obtenidos por la matriz, no indica que las clases con mayor confusión son paper y plastic, esto podría deberse a que encuentra similitudes en algunos casos.

```txt
[[162 1 1 1 7 5]
[ 1 134 5 0 2 11]
[ 4 8 159 1 1 4]
[ 2 0 2 154 11 8]
[ 6 0 5 14 189 5]
[ 6 16 11 5 9 189]]
```


- Métricas por clase:
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

## 8. Implementación en el NAO

Video de pruebas con el NAO: https://www.youtube.com/watch?v=rL4mmIgELGg
