# 🎹 NotasTT - Sistema de Clasificación y Generación de Notas Musicales

Un pipeline completo de Machine Learning para la clasificación automática de notas musicales de piano utilizando técnicas avanzadas de procesamiento de señales y Random Forest.

## 📋 Descripción General

**NotasTT** es un sistema de tres fases que permite:
1. **Generar datasets sintéticos** de notas de piano con variabilidad controlada
2. **Entrenar clasificadores** robustos usando características WPT (Wavelet Packet Transform)
3. **Realizar inferencia en tiempo real** sobre archivos de audio

### 🎯 Características Principales

- ✅ **Generación automática de datasets** con 88 notas (A0-C8)
- ✅ **Extracción de características avanzadas** usando WPT optimizado
- ✅ **Clasificadores ML entrenados** (Random Forest, Linear SVC, KNN, LogReg)
- ✅ **Inferencia con detección automática** de notas en audio
- ✅ **Análisis de confusión** por nota y octava
- ✅ **Pipeline completo** de preprocesamiento y normalización

---

## 🚀 Inicio Rápido

### 1️⃣ Instalación de Dependencias

#### FluidSynth (Sintetizador de audio)
```bash
# macOS
brew install fluidsynth

# Windows
choco install fluidsynth

# Ubuntu/Debian
sudo apt-get install fluidsynth
```

#### Dependencias Python
```bash
pip install -r requeriments.txt
```

**Dependencias incluidas:**
- `numpy`, `pandas` - Manejo de datos
- `scikit-learn` - Machine Learning
- `PyWavelets` - Wavelet Packet Transform
- `pretty_midi`, `mido` - Generación MIDI
- `soundfile` - Lectura/escritura de audio
- `matplotlib` - Visualización

### 2️⃣ Descargar SoundFont

Descarga `piano.sf2` desde:
**[Acoustic Grand Piano SoundFont](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)**

Colócalo en el directorio raíz del proyecto.

---

## 📁 Estructura del Proyecto

```
NotasTT/
├── GenerarNnotas.py         # Generador de N notas aleatorias
├── piano.sf2                # SoundFont (descargar aparte)
├── requeriments.txt         # Dependencias Python
│
├── phase1/                  # 🔹 FASE 1: Dataset básico
│   ├── GenerarNotas.py         # Generar dataset con variaciones
│   ├── DividirAudios.py        # Split train/valid/test
│   ├── ExtraerCaracteristicas.py  # WPT features
│   ├── EntrenarClasificador.py    # Entrenar modelos ML
│   ├── inferencia.py            # Inferencia sobre audios
│   ├── data/                    # Audios generados
│   ├── features_wpt_optimized/  # Features extraídas
│   ├── models_improved/         # Modelos entrenados
│   └── metadata/                # Metadatos CSV
│
├── phase2/                  # 🔹 FASE 2: Dataset robusto
│   ├── GenerarNotas.py         # 4 articulaciones + expresividad
│   ├── dividirAudio.py         # Split estratificado
│   ├── ExtraerCaracteristicas.py
│   ├── EntrenarClasificador.py
│   ├── Inferencia.py
│   ├── piano/                  # Dataset completo
│   ├── features/               # Features WPT
│   └── models/                 # Modelos entrenados
│
├── phase3/                  # 🔹 FASE 3: Generación de acordes
│   └── GenerarAcordesDS.py     # Acordes de 3 notas (tríadas)
│
└── demo_out/                # Demos y ejemplos generados
```

---

## � FASE 1: Pipeline Básico

### 1. Generar Dataset
```bash
cd phase1
python GenerarNotas.py
```

**Genera:**
- 88 notas (A0 a C8)
- 4 velocidades: 30, 60, 90, 110
- 2 articulaciones: staccato (0.3s), sustain (1.5s)
- 2 configuraciones de pedal
- **Total: 704 archivos WAV**

### 2. Dividir en Train/Valid/Test
```bash
python DividirAudios.py
```

**Distribución:**
- Train: 70%
- Valid: 15%
- Test: 15%

### 3. Extraer Características
```bash
python ExtraerCaracteristicas.py
```

**Características WPT optimizadas:**
- Descomposición Wavelet Packet (nivel 4)
- Estadísticas por banda: media, std, energía
- Normalización StandardScaler
- Dimensionalidad reducida

### 4. Entrenar Clasificadores
```bash
python EntrenarClasificador.py
```

**Modelos entrenados:**
- `RandomForest` (mejor rendimiento)
- `LinearSVC`
- `KNN`
- `LogisticRegression`

**Salida:**
- Modelos `.pkl` en `models_improved/`
- Matriz de confusión
- Reporte de clasificación
- Métricas por clase

### 5. Inferencia
```bash
python inferencia.py --wav ruta/al/audio.wav
```

**Funcionalidades:**
- Detección automática de onset
- Segmentación de notas
- Clasificación con probabilidades
- Generación de MIDI

---

## 🎵 FASE 2: Dataset Robusto

### Mejoras sobre Fase 1:
- ✅ **4 articulaciones**: staccato, portato, sustain, legato
- ✅ **Expresividad mejorada**: timing natural, dinámica variable
- ✅ **Normalización avanzada**: picos controlados, fade-in/fade-out
- ✅ **Total: 2,816 archivos**

### Pipeline Completo
```bash
cd phase2

# 1. Generar dataset robusto
python GenerarNotas.py

# 2. Dividir datos
python dividirAudio.py

# 3. Extraer features
python ExtraerCaracteristicas.py

# 4. Entrenar modelos
python EntrenarClasificador.py

# 5. Inferencia
python Inferencia.py --wav audio.wav
```

---

## 🎹 FASE 3: Generación de Acordes

### Acordes de 3 Notas (Tríadas)
```bash
cd phase3
python GenerarAcordesDS.py
```

**Tipos de acordes:**
- 20 acordes mayores (C_major, G_major, D_major...)
- 15 acordes menores (A_minor, E_minor, D_minor...)
- 10 acordes suspendidos (Csus2, Gsus4...)
- 5 acordes aumentados y disminuidos

**Configuración:**
- 2 velocidades: 60, 90
- Articulación: sustain (2.0s)
- Con pedal para sonido natural

---

## 🛠️ Utilidades Adicionales

### Generador de N Notas Aleatorias
```bash
# Generar 10 notas aleatorias
python GenerarNnotas.py -n 10 --normalize

# Con rango específico (C3 a C6)
python GenerarNnotas.py -n 8 --min_midi 48 --max_midi 84

# Sin repetición
python GenerarNnotas.py -n 12 --unique --seed 42

# Con separación personalizada
python GenerarNnotas.py -n 5 --gap 1.0 --dur 1.5
```

**Argumentos:**
- `-n, --num_notes`: Número de notas a generar (REQUERIDO)
- `--min_midi` / `--max_midi`: Rango MIDI (default: 21-108)
- `--unique`: No repetir notas
- `--seed`: Semilla para reproducibilidad
- `--gap`: Separación entre notas (default: 0.5s)
- `--dur`: Duración de cada nota (default: 1.0s)
- `--normalize`: Normalizar audio

---

## 📊 Formato de Metadatos

Cada fase genera un archivo CSV con metadatos detallados:

```csv
filepath,instrument,note,midi,velocity,articulation,pedal,seconds,fs,bits,peak_dbfs,source,soundfont
data/piano/C4/C4_v60_sustain_noped_01.wav,piano,C4,60,60,sustain,0,2.0,44100,16,-1.94,synth_sf2,piano.sf2
```

**Columnas:**
- `filepath`: Ruta al archivo WAV
- `note`: Nombre de la nota (C4, A#0, etc.)
- `midi`: Número MIDI (21-108)
- `velocity`: Velocidad MIDI (30-110)
- `articulation`: staccato/portato/sustain/legato
- `pedal`: 0 (sin) / 1 (con)
- `peak_dbfs`: Nivel de pico en dBFS

---

## 🧠 Arquitectura del Sistema

### Extracción de Características (WPT)
```python
# Wavelet Packet Transform optimizado
- Wavelets: db4, sym4
- Niveles: 4
- Estadísticas por banda:
  * Media, desviación estándar
  * Energía normalizada
  * Coeficientes de mayor magnitud
```

### Preprocesamiento de Audio
```python
1. Resample a 44.1kHz
2. Conversión a mono
3. Normalización de volumen
4. Detección de onset (phase2)
5. Segmentación automática
6. Feature extraction WPT
7. Normalización StandardScaler
```

### Clasificadores
```python
RandomForest:
  - n_estimators: 200
  - max_depth: 30
  - min_samples_split: 5
  - class_weight: balanced

LinearSVC:
  - kernel: linear
  - C: 1.0
  - max_iter: 5000
```

---

## 📈 Resultados Esperados

### Phase 1 (Dataset Básico)
- **Accuracy test**: ~95-98%
- **F1-score promedio**: ~0.96
- Mejores resultados en notas medias (C3-C5)

### Phase 2 (Dataset Robusto)
- **Accuracy test**: ~97-99%
- **F1-score promedio**: ~0.98
- Mejor generalización con articulaciones variadas

### Análisis de Confusión
- Mayor confusión en notas adyacentes (+/- 1 semitono)
- Octavas extremas (A0-C1, C7-C8) más desafiantes
- Acordes armónicos pueden causar confusión

---

## 🎯 Casos de Uso

### 1. **Transcripción automática de piano**
```bash
python phase2/Inferencia.py --wav recording.wav
# Genera: MIDI con notas detectadas
```

### 2. **Análisis de interpretaciones**
- Detectar velocidades de ejecución
- Identificar patrones rítmicos
- Comparar interpretaciones

### 3. **Entrenamiento de modelos ML**
- Dataset etiquetado de alta calidad
- Variabilidad controlada
- Metadatos completos

### 4. **Educación musical**
- Ejemplos de todas las notas
- Referencias de articulaciones
- Material didáctico

---

## 🔧 Configuración Avanzada

### Ajustar parámetros de generación (phase1/GenerarNotas.py)
```python
VELOCITIES = [30, 60, 90, 110]
ARTICULATIONS = {"staccato": 0.30, "sustain": 1.50}
PEDALS = {"noped": 0, "ped": 127}
MIDI_MIN, MIDI_MAX = 21, 108  # A0 a C8
```

### Modificar extracción de features
```python
# ExtraerCaracteristicas.py
WAVELETS = ['db4', 'sym4']
WPT_LEVEL = 4
SAMPLE_RATE = 44100
```

### Optimizar clasificadores
```python
# EntrenarClasificador.py
rf_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 5
}
```

---

## � Troubleshooting

### FluidSynth no encontrado
```bash
# Verificar instalación
fluidsynth --version

# Reinstalar si es necesario
brew reinstall fluidsynth  # macOS
```

### Error de versiones sklearn
```bash
# Actualizar scikit-learn
pip install --upgrade scikit-learn

# O reentrenar con versión actual
python EntrenarClasificador.py
```

### Audio distorsionado
- Verificar ganancia en `render_with_fluidsynth()`
- Ajustar parámetro `-g` de FluidSynth
- Usar `--normalize` en generación

### Accuracy baja
- Verificar que el dataset esté balanceado
- Aumentar variabilidad (velocidades, articulaciones)
- Probar diferentes wavelets en WPT
- Incrementar `n_estimators` en RandomForest

---

## 📚 Referencias

- **FluidSynth**: [fluidsynth.org](https://www.fluidsynth.org/)
- **SoundFont**: [freepats.zenvoid.org](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)
- **Wavelet Packet Transform**: PyWavelets documentation
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org/)

---

## � Autor

**Trabajo Terminal - ESCOM IPN**
- Repositorio: [github.com/TorresAbonceLuis/NotasTT](https://github.com/TorresAbonceLuis/NotasTT)

---

## 📄 Licencia

Este proyecto está bajo licencia MIT. El SoundFont utilizado puede tener su propia licencia.

---

## 🎵 ¡Disfruta clasificando notas musicales con ML! 🎹
