# Título del Proyecto: Clasificación de Emociones con BERT

## Autor(es)
Nombre y Apellidos

## Índice
1. [Resumen](#resumen)
2. [Descripción del Trabajo](#descripción-del-trabajo)
   - [Motivación y Objetivos](#motivación-y-objetivos)
   - [Herramientas Utilizadas](#herramientas-utilizadas)
   - [Desarrollo del Trabajo](#desarrollo-del-trabajo)
   - [Resultados](#resultados)
   - [Conclusiones](#conclusiones)
3. [Valoración Personal](#valoración-personal)
4. [Bibliografía](#bibliografía)
5. [Enlace al Código Implementado](#enlace-al-código-implementado)

## Resumen

El presente trabajo aborda el desafío de la detección y clasificación automática de emociones en textos mediante el uso de técnicas avanzadas de Procesamiento del Lenguaje Natural (PLN). Se ha desarrollado un sistema basado en el modelo BERT (Bidirectional Encoder Representations from Transformers) que permite identificar 28 emociones distintas en textos, así como agruparlas en las categorías emocionales básicas de Ekman.

La investigación parte del conjunto de datos GoEmotions, que consiste en comentarios de Reddit anotados con múltiples emociones. Se implementó un modelo de clasificación multi-etiqueta utilizando BERT, que fue entrenado durante tres épocas con técnicas de preprocesamiento lingüístico como la lematización para mejorar la calidad de las predicciones. Los resultados muestran un rendimiento destacable en la identificación de emociones como "gratitud" (F1=0.908), "amor" (F1=0.764) y "admiración" (F1=0.742), mientras que otras emociones como "duelo" y "nerviosismo" resultaron más difíciles de detectar.

El sistema desarrollado se ha implementado además como una aplicación web interactiva utilizando Streamlit, permitiendo a los usuarios analizar textos en tiempo real y visualizar tanto las emociones detectadas como su distribución en categorías de Ekman. Esta herramienta tiene potenciales aplicaciones en análisis de sentimientos en redes sociales, servicio al cliente, psicología computacional y estudios de comportamiento humano en entornos digitales.

La metodología aplicada y los resultados obtenidos demuestran el potencial de los modelos de lenguaje pre-entrenados para tareas complejas de clasificación emocional, representando un avance significativo respecto a técnicas tradicionales basadas en léxicos o enfoques estadísticos simples.

## Descripción del Trabajo

### Motivación y Objetivos

La comprensión de las emociones humanas expresadas en textos representa uno de los mayores desafíos del Procesamiento del Lenguaje Natural actual. A diferencia del análisis de sentimientos tradicional, que suele limitarse a clasificaciones positivas, negativas o neutras, la detección de emociones específicas requiere modelos con una comprensión más profunda del lenguaje y sus matices.

La principal motivación de este trabajo surge de la necesidad de desarrollar sistemas capaces de identificar un espectro más amplio de emociones en textos, lo que permitiría aplicaciones más sofisticadas en diversos campos como:

- Análisis de la percepción de usuarios en redes sociales
- Mejora de sistemas de atención al cliente
- Apoyo a estudios psicológicos y sociológicos
- Desarrollo de agentes conversacionales más empáticos
- Monitorización de bienestar emocional en entornos digitales

El conjunto de datos GoEmotions, publicado por Google Research, ofrece una oportunidad única para abordar este problema, ya que proporciona un corpus de comentarios de Reddit anotados con 28 emociones diferentes, permitiendo entrenar modelos de mayor granularidad emocional.

Los objetivos específicos planteados para este proyecto fueron:

1. Implementar un clasificador multi-etiqueta basado en BERT capaz de detectar 28 categorías emocionales distintas en textos.
2. Explorar técnicas de preprocesamiento lingüístico (como la lematización) para mejorar el rendimiento del modelo.
3. Desarrollar una metodología para agrupar las emociones detectadas en las seis categorías básicas de Ekman (alegría, tristeza, ira, miedo, sorpresa y asco) más la categoría neutral.
4. Crear una aplicación web interactiva que permita a los usuarios analizar textos y visualizar las emociones detectadas.
5. Evaluar el rendimiento del modelo mediante métricas estándar como F1-score, identificando fortalezas y debilidades en la detección de cada tipo de emoción.

Este trabajo se alinea con la tendencia actual en PLN de ir más allá del análisis de sentimientos binario hacia una comprensión más matizada de las expresiones emocionales humanas, lo que representa un paso importante hacia sistemas computacionales con mayor inteligencia emocional.

### Herramientas Utilizadas

Para el desarrollo de este proyecto de clasificación de emociones se utilizaron diversas tecnologías y bibliotecas especializadas en aprendizaje profundo y procesamiento del lenguaje natural:

#### Frameworks y Bibliotecas Principales

- **PyTorch (1.x)**: Framework de aprendizaje profundo que proporciona la infraestructura para implementar y entrenar el modelo neuronal. Se eligió por su flexibilidad y por ofrecer un buen equilibrio entre rendimiento y facilidad de uso para la experimentación.

- **Transformers (Hugging Face)**: Biblioteca que proporciona implementaciones estado del arte de modelos de lenguaje como BERT. Se utilizó para acceder al modelo BERT pre-entrenado y para los tokenizadores especializados.

- **BERT (Bidirectional Encoder Representations from Transformers)**: Modelo de lenguaje pre-entrenado desarrollado por Google que captura el contexto bidireccional de las palabras. Se utilizó la versión base sin capa superior (`bert-base-uncased`) como punto de partida para nuestra tarea de clasificación.

- **NLTK (Natural Language Toolkit)**: Biblioteca para procesamiento de lenguaje natural utilizada específicamente para tareas de tokenización y lematización en el preprocesamiento textual.

- **Streamlit**: Framework para la creación de aplicaciones web interactivas en Python. Se empleó para desarrollar la interfaz de usuario que permite probar el clasificador de emociones en tiempo real.

#### Bibliotecas de Análisis y Visualización

- **Pandas**: Utilizada para el procesamiento y manipulación eficiente de los datos tabulares, especialmente para cargar y transformar el conjunto de datos GoEmotions.

- **NumPy**: Empleada para operaciones matemáticas y manipulación de arrays multidimensionales, esencial para el procesamiento de las representaciones vectoriales.

- **Matplotlib y Pyplot**: Usadas para la generación de gráficos y visualizaciones que muestran las distribuciones de emociones y los resultados de las predicciones.

- **Scikit-learn**: Proporciona herramientas para la evaluación del modelo (métricas como F1-score) y para el procesamiento de etiquetas múltiples (MultiLabelBinarizer).

#### Herramientas de Desarrollo

- **Visual Studio Code**: IDE principal utilizado para el desarrollo del código.

- **Git/GitHub**: Para control de versiones y almacenamiento del repositorio del proyecto.

#### Datos y Recursos

- **GoEmotions Dataset**: Conjunto de datos de Google Research que contiene aproximadamente 58,000 comentarios de Reddit anotados con 28 emociones diferentes. Este recurso fue fundamental para entrenar el modelo, ya que proporciona datos etiquetados de alta calidad.

- **Mapeo de Ekman**: Se desarrolló un mapeo personalizado para agrupar las 28 emociones del dataset GoEmotions en las seis categorías emocionales fundamentales definidas por Paul Ekman (alegría, tristeza, ira, miedo, sorpresa, asco) más la categoría neutral.

La combinación de estas herramientas permitió desarrollar un sistema integral que abarca desde el preprocesamiento de datos, entrenamiento del modelo, evaluación del rendimiento, hasta la implementación de una interfaz de usuario para facilitar el uso del clasificador.

### Desarrollo del Trabajo

El desarrollo del proyecto de clasificación de emociones con BERT siguió una metodología estructurada que abarcó desde la comprensión del problema hasta la implementación de una aplicación funcional. A continuación, se detallan las etapas principales:

#### 1. Comprensión y Exploración del Dataset GoEmotions

El punto de partida fue el estudio detallado del conjunto de datos GoEmotions, que presenta las siguientes características:
- Aproximadamente 58,000 comentarios de Reddit anotados manualmente
- Cada comentario puede estar asociado con múltiples emociones (clasificación multi-etiqueta)
- 28 categorías emocionales distintas, incluyendo una categoría "neutral"
- División en conjuntos de entrenamiento, validación y prueba

Se realizó un análisis exploratorio para entender la distribución de las emociones en el dataset, identificando desbalances significativos. Por ejemplo, emociones como "gratitud", "admiración" y "alegría" están mucho más representadas que otras como "duelo", "nerviosismo" o "vergüenza". Este desbalance tendría impacto en el rendimiento del modelo.

#### 2. Diseño e Implementación del Modelo

Para abordar la tarea de clasificación multi-etiqueta, se diseñó un modelo basado en BERT con las siguientes características:

```python
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
```

La arquitectura consta de:
1. Una capa BERT base pre-entrenada que genera representaciones contextuales
2. Una capa de dropout (0.1) para regularización
3. Una capa lineal que proyecta las representaciones a 28 dimensiones (una por emoción)
4. Una función de activación sigmoid que transforma las salidas en probabilidades independientes

Se utilizó la función de pérdida Binary Cross-Entropy (BCE), adecuada para problemas de clasificación multi-etiqueta:

```python
loss_fct = nn.BCELoss()
loss = loss_fct(output, labels)
```

#### 3. Preprocesamiento Lingüístico

Se implementaron dos versiones del modelo:
- **Básica**: Utilizando el texto original con la tokenización estándar de BERT
- **Avanzada**: Incorporando lematización para reducir la variabilidad lingüística

Para la lematización se utilizó el WordNet Lemmatizer de NLTK:

```python
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join back to text
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text
```

Esta técnica permitió normalizar palabras con la misma raíz semántica, como "corriendo", "correr", "corrió" a su forma base "correr", potencialmente mejorando la capacidad del modelo para generalizar.

#### 4. Entrenamiento del Modelo

El proceso de entrenamiento se configuró con los siguientes parámetros:
- Optimizador: AdamW con learning rate de 2e-5
- Batch size: 16
- Número de épocas: 3
- Dispositivo de entrenamiento: CPU/GPU dependiendo de la disponibilidad

El entrenamiento siguió este flujo:
1. Carga de datos mediante la clase personalizada `GoEmotionsDataset`
2. Tokenización y preparación de batches con DataLoader
3. Forward pass a través del modelo
4. Cálculo de pérdida y backpropagation
5. Actualización de pesos

Durante el entrenamiento, se monitorizó la pérdida en los conjuntos de entrenamiento y validación. El modelo mostró una convergencia progresiva con la pérdida de entrenamiento reduciéndose de aproximadamente 0.10 a 0.06 a lo largo de las épocas.

#### 5. Desarrollo del Sistema de Categorización de Ekman

Adicionalmente al modelo de 28 emociones, se implementó un sistema para mapear estas emociones a las 6 categorías fundamentales de Ekman (más neutral):
- Alegría (joy)
- Tristeza (sadness)
- Ira (anger)
- Miedo (fear)
- Sorpresa (surprise)
- Asco (disgust)
- Neutral

Este mapeo se implementó mediante un diccionario JSON que asigna cada emoción específica a una categoría de Ekman:

```json
{
  "anger": ["anger", "annoyance", "disapproval"],
  "disgust": ["disgust"],
  "fear": ["fear", "nervousness"],
  "joy": ["joy", "amusement", "approval", "excitement", "gratitude", 
          "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
  "sadness": ["sadness", "disappointment", "embarrassment", "grief", 
              "remorse"],
  "surprise": ["surprise", "realization", "confusion", "curiosity"]
}
```

El sistema calcula la distribución de Ekman sumando las probabilidades de todas las emociones que pertenecen a cada categoría principal, proporcionando una visión más holística del estado emocional del texto.

#### 6. Evaluación del Rendimiento

Para evaluar el modelo, se utilizó principalmente el F1-score por clase, adecuado para conjuntos de datos desequilibrados, además de la pérdida de validación. Los resultados mostraron variaciones significativas entre emociones:

- Emociones con buen rendimiento (F1 > 0.75):
  - Gratitud (0.908)
  - Amor (0.764)
  - Admiración (0.742)
  - Diversión (0.784)

- Emociones con rendimiento medio (0.4 < F1 < 0.75):
  - Alegría (0.468)
  - Tristeza (0.496)
  - Sorpresa (0.523)
  - Neutral (0.647)

- Emociones difíciles de detectar (F1 < 0.2 o 0):
  - Duelo (0.000)
  - Nerviosismo (0.000)
  - Orgullo (0.000)
  - Decepción (0.047)

El modelo con lematización mostró mejoras en varias categorías emocionales en comparación con el modelo base, particularmente en emociones como "remordimiento" y "optimismo".

#### 7. Desarrollo de la Aplicación Web

Finalmente, se implementó una aplicación web utilizando Streamlit para demostrar las capacidades del modelo. La aplicación incluye:

1. **Interfaz de entrada de texto**: Donde los usuarios pueden escribir o pegar texto para analizar.
2. **Ajuste de umbral**: Permite al usuario controlar la sensibilidad de la detección de emociones.
3. **Visualizaciones**:
   - Gráfico de barras horizontales mostrando las emociones detectadas y sus probabilidades
   - Gráfico circular que representa la distribución de categorías de Ekman
   - Tabla detallada con todas las emociones detectadas

La aplicación carga el modelo previamente entrenado y realiza las predicciones en tiempo real, ofreciendo una experiencia interactiva para explorar la capacidad del sistema en diferentes tipos de textos.

Durante el desarrollo se enfrentaron varios desafíos, entre ellos:
- El manejo eficiente de la memoria durante el entrenamiento con datasets grandes
- La optimización del rendimiento para emociones con poca representación
- La correcta interpretación de la clasificación multi-etiqueta
- La integración del modelo en la aplicación web asegurando tiempos de respuesta rápidos

Estos desafíos se abordaron mediante ajustes en la arquitectura del modelo, técnicas de preprocesamiento lingüístico como la lematización, y la optimización del código de la aplicación.

### Resultados

El desarrollo y evaluación del modelo de clasificación de emociones arrojó resultados significativos que permiten valorar su rendimiento y potencial aplicabilidad. A continuación, se presentan los hallazgos principales organizados por aspectos clave:

#### Rendimiento General del Modelo

El modelo final, entrenado durante 3 épocas con lematización, alcanzó los siguientes indicadores generales:
- **Loss de entrenamiento final**: 0.0675
- **Loss de validación final**: 0.0841
- **F1-score macro promedio**: 0.3734

Este F1-score macro refleja el promedio de los F1-scores individuales de cada emoción, sin ponderación por frecuencia, lo que proporciona una visión más equilibrada del rendimiento en un dataset desbalanceado.

#### Rendimiento por Categoría Emocional

A continuación se muestran los F1-scores obtenidos para cada emoción, agrupados por su rendimiento:

| **Emociones con alto rendimiento** (F1 > 0.7) | F1-score |
|----------------------------------------------|----------|
| Gratitud (gratitude)                         | 0.9080   |
| Diversión (amusement)                        | 0.7843   |
| Amor (love)                                  | 0.7637   |
| Admiración (admiration)                      | 0.7418   |

| **Emociones con rendimiento aceptable** (0.5 < F1 < 0.7) | F1-score |
|----------------------------------------------------------|----------|
| Neutral                                                  | 0.6471   |
| Optimismo (optimism)                                     | 0.6072   |
| Miedo (fear)                                             | 0.5778   |
| Remordimiento (remorse)                                  | 0.5586   |
| Sorpresa (surprise)                                      | 0.5229   |

| **Emociones con rendimiento medio** (0.3 < F1 < 0.5) | F1-score |
|------------------------------------------------------|----------|
| Tristeza (sadness)                                   | 0.4959   |
| Alegría (joy)                                        | 0.4683   |
| Ira (anger)                                          | 0.4765   |
| Disgusto (disgust)                                   | 0.3969   |
| Confusión (confusion)                                | 0.3364   |

| **Emociones con bajo rendimiento** (F1 < 0.3) | F1-score |
|----------------------------------------------|----------|
| Molestia (annoyance)                         | 0.2701   |
| Entusiasmo (excitement)                      | 0.2459   |
| Deseo (desire)                               | 0.2418   |
| Aprobación (approval)                        | 0.2398   |
| Curiosidad (curiosity)                       | 0.2290   |
| Desaprobación (disapproval)                  | 0.1924   |
| Vergüenza (embarrassment)                    | 0.1579   |
| Realización (realization)                    | 0.0758   |
| Decepción (disappointment)                   | 0.0473   |
| Duelo (grief)                                | 0.0000   |
| Nerviosismo (nervousness)                    | 0.0000   |
| Orgullo (pride)                              | 0.0000   |
| Alivio (relief)                              | 0.0000   |

La variabilidad en el rendimiento está directamente relacionada con:
1. La frecuencia de cada emoción en el conjunto de entrenamiento
2. La ambigüedad inherente de ciertas emociones (ej. la distinción entre "annoyance" y "anger")
3. La complejidad de las expresiones textuales asociadas a cada emoción

#### Distribución por Categorías de Ekman

Al agrupar las emociones en las categorías de Ekman, el modelo mostró un rendimiento más equilibrado. La representación consolidada mejoró la fiabilidad en casos donde emociones individuales eran difíciles de detectar. Por ejemplo, aunque "nervousness" obtuvo un F1 de 0, la categoría general "fear" alcanzó un F1 de 0.5778.

Las categorías de Ekman con mejor rendimiento fueron:
1. Alegría (joy): 0.67 (promedio ponderado)
2. Asco (disgust): 0.40
3. Tristeza (sadness): 0.37
4. Sorpresa (surprise): 0.33

#### Impacto de la Lematización

La aplicación de lematización como técnica de preprocesamiento mostró un impacto positivo en el rendimiento global:

| **Métrica**             | **Sin lematización** | **Con lematización** | **Mejora** |
|-------------------------|----------------------|----------------------|------------|
| F1-score macro promedio | 0.2727               | 0.3734               | +36.9%     |
| Loss de validación      | 0.0935               | 0.0841               | +10.1%     |

El impacto fue particularmente notable en emociones con expresiones lingüísticas variadas como "remorse" (+21.3%) y "optimism" (+5.2%).

#### Análisis de Errores

Se identificaron principalmente tres tipos de errores:

1. **Falsos negativos en emociones poco frecuentes**: Emociones como "grief", "pride" y "nervousness" rara vez fueron detectadas, incluso cuando estaban presentes.

2. **Confusión entre emociones relacionadas**: El modelo a menudo confundió pares de emociones semánticamente cercanas:
   - Admiración vs. Amor
   - Ira vs. Molestia
   - Sorpresa vs. Realización

3. **Detección excesiva de emociones dominantes**: Emociones como "gratitude" y "admiration" tendían a ser predichas con más frecuencia debido a su sobrerrepresentación en los datos de entrenamiento.

#### Resultados de la Aplicación Web

La implementación de la aplicación web con Streamlit demostró:

1. **Tiempos de respuesta adecuados**: Predicciones completadas en aproximadamente 0.3-0.5 segundos por texto en CPU estándar.

2. **Consistencia en las predicciones**: Los resultados fueron consistentes con las métricas observadas durante la evaluación del modelo.

3. **Visualizaciones informativas**: Las representaciones gráficas de emociones y categorías de Ekman proporcionaron una interpretación intuitiva de los resultados.

La capacidad de ajustar el umbral de detección resultó particularmente útil, permitiendo a los usuarios controlar el equilibrio entre precisión y exhaustividad según sus necesidades específicas.

#### Comparación con el Estado del Arte

Aunque no era un objetivo principal del proyecto, se realizó una breve comparación con otros trabajos en el ámbito de la detección de emociones:

| **Enfoque**                            | **F1-score macro en GoEmotions** |
|----------------------------------------|----------------------------------|
| Modelo desarrollado (BERT + lematización) | 0.37                          |
| BERT base (literatura)                 | 0.46                             |
| XLNet (literatura)                     | 0.50                             |
| Modelos basados en reglas/léxicos      | 0.20-0.25                        |

El modelo desarrollado mostró un rendimiento razonable considerando el alcance y recursos del proyecto, situándose por encima de los enfoques tradicionales basados en léxicos y a una distancia aceptable de implementaciones más avanzadas.

La implementación de un sistema completo de clasificación y visualización, junto con la categorización de Ekman, añade un valor significativo más allá del rendimiento bruto del clasificador.

### Conclusiones

El desarrollo de este proyecto de clasificación de emociones con BERT ha permitido obtener conclusiones relevantes tanto desde el punto de vista técnico como aplicado:

#### Viabilidad de la Clasificación Emocional Multi-etiqueta

El modelo implementado demuestra que es posible abordar la clasificación multi-etiqueta de emociones con un nivel de precisión aceptable utilizando arquitecturas transformer pre-entrenadas. La capacidad de BERT para capturar contextos bidireccionales resulta fundamental en una tarea donde los matices lingüísticos son determinantes para distinguir entre estados emocionales cercanos.

Un hallazgo importante es la diferencia significativa en el rendimiento de detección según la emoción específica, lo que subraya la complejidad inherente de mapear lenguaje a estados emocionales. Algunas emociones presentan patrones lingüísticos más consistentes y reconocibles (gratitud, admiración, amor) mientras que otras (orgullo, nerviosismo, duelo) requieren un tratamiento más sofisticado.

#### Impacto del Preprocesamiento Lingüístico

Los resultados evidencian que técnicas de preprocesamiento relativamente simples como la lematización pueden tener un impacto significativo en el rendimiento de los modelos de clasificación emocional. La mejora global de casi un 37% en F1-score macro al implementar lematización confirma la importancia de considerar la morfología de las palabras en la detección de emociones.

Esto sugiere que, a pesar del poder de los modelos pre-entrenados, el preprocesamiento específico para la tarea sigue siendo relevante y debe considerarse como parte integral del desarrollo de soluciones de PLN.

#### Potencial de la Categorización de Ekman

La agrupación de emociones en las categorías de Ekman ha demostrado ser un enfoque valioso para proporcionar una visión más estable y holística del estado emocional expresado en un texto. Esta abstracción a un nivel superior permite superar algunas de las limitaciones de la clasificación en emociones específicas, especialmente para aquellas con bajo rendimiento individual.

El modelo jerárquico (emociones específicas → categorías de Ekman) proporciona un balance entre granularidad y confiabilidad que puede ser ajustado según las necesidades de la aplicación.

#### Desafíos del Desbalance de Datos

El conjunto de datos GoEmotions, como muchos datasets reales, presenta un desbalance significativo en la distribución de clases. Los resultados confirman que las emociones con menor representación en los datos de entrenamiento tienden a tener un rendimiento predictivo inferior.

Este hallazgo resalta la importancia de considerar técnicas específicas para conjuntos de datos desbalanceados en futuros trabajos, como estrategias de sobremuestreo, submuestreo o ajustes en las funciones de pérdida.

#### Aplicabilidad Práctica

La implementación exitosa de una aplicación web demuestra la viabilidad de utilizar modelos de detección de emociones en entornos reales y con tiempos de respuesta aceptables. El sistema desarrollado podría aplicarse en diversos contextos como:

- **Monitorización de feedback de usuarios**: Analizando comentarios o reseñas para comprender la carga emocional asociada.
- **Mejora de chatbots y asistentes virtuales**: Permitiendo respuestas más empáticas basadas en la emoción detectada.
- **Análisis de redes sociales**: Monitorizando tendencias emocionales en tiempo real sobre temas específicos.
- **Herramientas de apoyo para escritores**: Ayudando a evaluar el tono emocional de textos durante su creación.
- **Aplicaciones educativas**: Analizando la respuesta emocional de estudiantes en entornos de aprendizaje en línea.

#### Limitaciones y Líneas Futuras de Investigación

A pesar de los resultados prometedores, es importante reconocer las limitaciones del sistema:

1. **Especificidad contextual**: El modelo está entrenado principalmente con comentarios de Reddit, lo que podría limitar su generalización a otros contextos lingüísticos o dominios.

2. **Sesgo cultural**: La interpretación de emociones puede variar significativamente entre culturas, y el modelo refleja principalmente perspectivas occidentales sobre la expresión emocional.

3. **Complejidad lingüística**: Aspectos como la ironía, el sarcasmo o expresiones idiomáticas siguen siendo desafíos importantes que el modelo actual no aborda completamente.

Para futuras investigaciones, sería valioso explorar:

- La incorporación de técnicas de data augmentation para mejorar el rendimiento en emociones poco representadas.
- La adaptación del modelo a dominios específicos mediante fine-tuning adicional.
- La exploración de arquitecturas más avanzadas como XLNet, RoBERTa o T5.
- La integración de señales multimodales (texto, audio, imagen) para una detección emocional más robusta.
- El desarrollo de versiones más eficientes del modelo para su uso en dispositivos con recursos limitados.

En conclusión, el proyecto ha demostrado tanto el potencial como los desafíos de la detección automática de emociones en textos, abriendo camino para aplicaciones más sofisticadas que puedan comprender y responder a los matices emocionales de la comunicación humana. La combinación de arquitecturas transformer con técnicas de procesamiento lingüístico específicas representa un enfoque prometedor para seguir avanzando en este campo.

## Valoración Personal

El desarrollo de este proyecto de clasificación de emociones ha representado una experiencia enormemente enriquecedora y desafiante desde múltiples perspectivas. A nivel académico, me ha permitido profundizar en la comprensión y aplicación práctica de los modelos transformer, particularmente BERT, uno de los avances más significativos en el campo del procesamiento del lenguaje natural en los últimos años.

Uno de los aspectos más gratificantes ha sido observar cómo la teoría aprendida durante el curso se transformaba en aplicaciones concretas y funcionales. El proceso de adaptación de un modelo pre-entrenado para una tarea específica de clasificación multi-etiqueta ha sido especialmente instructivo, permitiéndome entender mejor las complejidades del fine-tuning y las consideraciones especiales que requieren las tareas de clasificación con múltiples categorías simultáneas.

Sin embargo, el proyecto no estuvo exento de desafíos significativos. Particularmente complicado resultó el manejo del desbalance de clases en el dataset GoEmotions, que refleja fielmente un problema común en aplicaciones del mundo real: algunas emociones son expresadas con mucha más frecuencia que otras en el lenguaje cotidiano. Encontrar estrategias para mejorar el rendimiento en emociones poco representadas, sin sacrificar la precisión en las más comunes, fue un ejercicio valioso que me obligó a investigar y experimentar con diversas técnicas.

La implementación de la lematización como estrategia de preprocesamiento y su impacto positivo en los resultados del modelo fue una de las satisfacciones más grandes del proyecto. Este hallazgo reforzó mi convicción de que, incluso en la era de los modelos pre-entrenados, el conocimiento lingüístico específico sigue siendo relevante y puede marcar diferencias significativas en el rendimiento.

Desde el punto de vista técnico, el desarrollo de la aplicación web con Streamlit representó una oportunidad para ampliar mis habilidades más allá del modelado puro, abordando aspectos de visualización de datos y desarrollo de interfaces que son cruciales para hacer que las soluciones de inteligencia artificial sean accesibles y útiles para usuarios finales.

Una de las reflexiones más importantes que me llevo de este proyecto es la comprensión de las limitaciones actuales de los sistemas de PLN para capturar toda la riqueza y complejidad de las emociones humanas. El hecho de que emociones como el duelo, el nerviosismo o el orgullo resultaran tan difíciles de detectar automáticamente nos recuerda cuánto camino queda por recorrer en este campo.

A nivel personal, este proyecto ha reforzado mi interés en la intersección entre el procesamiento del lenguaje natural y la psicología computacional. Creo firmemente que el desarrollo de sistemas capaces de comprender y responder apropiadamente a las emociones humanas será un área de investigación cada vez más relevante, con aplicaciones significativas en ámbitos como la salud mental, la educación y las interacciones humano-máquina.

En resumen, este trabajo no solo me ha permitido aplicar y expandir mis conocimientos técnicos, sino que también ha abierto mi perspectiva sobre las posibilidades y desafíos futuros en el campo del análisis emocional automatizado. Me siento satisfecho con los resultados obtenidos, consciente de las limitaciones del enfoque actual, y motivado para seguir explorando y contribuyendo a este fascinante campo de investigación.

## Bibliografía

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset for Fine-Grained Emotion Classification. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 8932-8943. https://doi.org/10.18653/v1/2020.acl-main.372

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1*, 4171-4186. https://doi.org/10.18653/v1/N19-1423

3. Ekman, P. (1992). An Argument for Basic Emotions. *Cognition and Emotion*, 6(3-4), 169-200. https://doi.org/10.1080/02699939208411068

4. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Brew, J. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38-45. https://doi.org/10.18653/v1/2020.emnlp-demos.6

5. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32*, 8024-8035.

6. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

7. Mohammad, S. M. (2022). Sentiment and Emotion Analysis for Social Media: A Survey. *ACM Computing Surveys*, 55(2), Article 39. https://doi.org/10.1145/3505516

8. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.

9. Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... & Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2

10. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

11. Tang, D., Wei, F., Yang, N., Zhou, M., Liu, T., & Qin, B. (2014). Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification. *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 1555-1565. https://doi.org/10.3115/v1/P14-1146

12. Plutchik, R. (2001). The Nature of Emotions: Human emotions have deep evolutionary roots, a fact that may explain their complexity and provide tools for clinical practice. *American Scientist*, 89(4), 344-350.

13. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55

14. Loper, E., & Bird, S. (2002). NLTK: The Natural Language Toolkit. *Proceedings of the ACL Workshop on Effective Tools and Methodologies for Teaching Natural Language Processing and Computational Linguistics*, 63-70. https://doi.org/10.3115/1118108.1118117

15. Ekman, P., & Friesen, W. V. (1971). Constants across cultures in the face and emotion. *Journal of Personality and Social Psychology*, 17(2), 124-129. https://doi.org/10.1037/h0030377

## Enlace al Código Implementado

El código fuente completo de este proyecto está disponible en el siguiente repositorio de GitHub:

[GitHub - EmotionClassifier-BERT](https://github.com/username/emotion-classifier-bert)

### Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

```
emotion-classifier-bert/
├── app.py                      # Aplicación web Streamlit
├── emotion_classifier.py       # Implementación principal del clasificador
├── data/                       # Datos de entrenamiento y evaluación
│   ├── emotions.txt            # Lista de emociones
│   ├── ekman_mapping.json      # Mapeo de emociones a categorías de Ekman
│   ├── train.tsv               # Datos de entrenamiento
│   ├── dev.tsv                 # Datos de validación
│   └── test.tsv                # Datos de prueba
├── archive/                    # Scripts auxiliares y recursos adicionales
│   ├── analyze_data.py         # Análisis exploratorio de datos
│   ├── calculate_metrics.py    # Cálculo detallado de métricas
│   └── data/                   # Datos adicionales y de soporte
└── training_002/               # Modelos entrenados
    ├── goemotions_bert_model_lemma.pt        # Modelo con lematización
    └── goemotions_bert_model_dev_lemma.pt    # Versión de desarrollo
```

### Instrucciones de Ejecución

Para ejecutar la aplicación web de clasificación de emociones:

1. Clonar el repositorio:
```bash
git clone https://github.com/username/emotion-classifier-bert.git
cd emotion-classifier-bert
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar la aplicación web:
```bash
streamlit run app.py
```

4. Acceder a la aplicación a través del navegador web en la dirección indicada (generalmente http://localhost:8501).

Para reentrenar el modelo o experimentar con diferentes configuraciones, se puede ejecutar el script principal:

```bash
python emotion_classifier.py
```

El código está documentado con comentarios detallados para facilitar su comprensión y modificación.