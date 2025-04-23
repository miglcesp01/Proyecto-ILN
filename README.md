# Proyecto ILN - Clasificador de Emociones

Este proyecto implementa un clasificador de emociones basado en el dataset GoEmotions, utilizando técnicas de procesamiento de lenguaje natural.

## Estructura del Proyecto

- `app.py`: Aplicación principal del proyecto
- `emotion_classifier.py`: Implementación del clasificador de emociones
- `data/`: Contiene los conjuntos de datos para entrenamiento, validación y prueba
- `archive/`: Utilidades de análisis y procesamiento de datos
- `results/`: Resultados de experimentos y ejecuciones
- `training_001/`: Modelos entrenados (primera versión)
- `training_002/`: Modelos entrenados con lematización (segunda versión)

## Modelos Entrenados

El proyecto incluye varios modelos pre-entrenados:
- Modelo base: `training_001/goemotions_bert_model.pt`
- Modelo con lematización: `training_002/goemotions_bert_model_lemma.pt`

## Resultados

Los resultados de los experimentos se pueden encontrar en la carpeta `results/runs/`, donde se guardan los logs de TensorBoard con métricas de rendimiento.

## Documentación

- `Memoria_del_trabajo.md/.tex`: Documentación completa del proyecto
- `PropuestaDeProyecto.docx/.pdf`: Propuesta inicial del proyecto

## Requisitos

Para ejecutar este proyecto, se recomienda:
1. Crear un entorno virtual de Python
2. Instalar las dependencias necesarias (próximamente se incluirá un archivo requirements.txt)

## Contribuciones

Este proyecto forma parte de un trabajo académico para la asignatura de ILN (Introducción al Lenguaje Natural).

---

© 2025 - Proyecto ILN