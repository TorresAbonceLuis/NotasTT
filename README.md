# 🎹 NotasTT - Generador de Dataset de Piano

Un generador automático de samples de piano que crea un dataset completo con todas las notas del piano usando FluidSynth y soundfonts.

## 📋 Descripción

Este proyecto genera automáticamente **704 archivos de audio** que cubren:
- **88 notas** del piano completo (A0 a C8)
- **4 velocidades** diferentes (30, 60, 90, 110)
- **2 articulaciones** (staccato 0.3s, sustain 1.5s)
- **2 configuraciones de pedal** (con y sin pedal de sustain)

Cada archivo está optimizado con:
- ✅ **Volumen normalizado** (90% del rango máximo)
- ✅ **Silencio final recortado** automáticamente
- ✅ **Audio de alta calidad** (44.1kHz, 16-bit WAV)
- ✅ **Metadatos completos** en formato CSV

## 🚀 Inicio Rápido

### 1. Dependencias

```bash
# macOS - Instalar FluidSynth
brew install fluidsynth

# Instalar librerías Python
pip install -r requeriments.txt
```

### 2. Descargar Soundfont

Descarga el archivo `piano.sf2` desde:
**[Acoustic Grand Piano SoundFont](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)**

Opciones disponibles:
- **SF2** (296 MiB) - Recomendado para uso general
- **SFZ + FLAC** (707 MiB) - Mejor calidad, 48kHz 24bit
- **SFZ + WAV** (1.18 GiB) - Máxima calidad, 48kHz 24bit
- **SFZ + WAV** (394 MiB) - Calidad estándar, 44.1kHz 16bit

Coloca el archivo `piano.sf2` en el directorio raíz del proyecto.

### 3. Ejecutar

```bash
python GenerarNotas.py
```

## 📁 Estructura del Proyecto

```
NotasTT/
├── GenerarNotas.py          # Script principal
├── piano.sf2                # SoundFont del piano (descargar aparte)
├── requeriments.txt         # Dependencias Python
├── README.md                # Este archivo
├── data/
│   └── piano_A/            # Archivos de audio generados
│       ├── A0/             # Carpeta por nota
│       │   ├── A0_v30_staccato_noped_01.mid
│       │   ├── A0_v30_staccato_noped_01.wav
│       │   └── ...
│       ├── A#0/
│       ├── B0/
│       └── ...             # Hasta C8
└── metadata/
    └── index.csv           # Metadatos de todos los archivos
```

## 🎵 Archivos Generados

Cada nota genera **8 variaciones**:

| Velocidad | Articulación | Pedal | Ejemplo |
|-----------|--------------|-------|---------|
| 30, 60, 90, 110 | staccato (0.3s) | sin pedal | `C4_v60_staccato_noped_01.wav` |
| 30, 60, 90, 110 | staccato (0.3s) | con pedal | `C4_v60_staccato_ped_01.wav` |
| 30, 60, 90, 110 | sustain (1.5s) | sin pedal | `C4_v60_sustain_noped_01.wav` |
| 30, 60, 90, 110 | sustain (1.5s) | con pedal | `C4_v60_sustain_ped_01.wav` |

### Nomenclatura
```
{Nota}_{Velocidad}_{Articulación}_{Pedal}_{Toma}.wav

Ejemplos:
- A0_v30_staccato_noped_01.wav    # La0, velocidad 30, staccato, sin pedal
- C4_v110_sustain_ped_01.wav      # Do4, velocidad 110, sustain, con pedal
- C8_v90_staccato_noped_01.wav    # Do8, velocidad 90, staccato, sin pedal
```

## 📊 Metadatos (CSV)

El archivo `metadata/index.csv` contiene información detallada de cada sample:

```csv
filepath,instrument,note,midi,velocity,articulation,pedal,seconds,fs,bits,peak_dbfs,source,soundfont
data/piano_A/C4/C4_v60_sustain_noped_01.wav,piano_A,C4,60,60,sustain,0,2.0,44100,16,-1.94,synth_sf2,piano.sf2
```

**Columnas:**
- `filepath`: Ruta al archivo WAV
- `instrument`: Tipo de instrumento
- `note`: Nombre de la nota (C4, A#0, etc.)
- `midi`: Número MIDI de la nota (21-108)
- `velocity`: Velocidad MIDI (30-110)
- `articulation`: Tipo de articulación (staccato/sustain)
- `pedal`: Pedal de sustain (0=sin, 1=con)
- `seconds`: Duración teórica del archivo
- `fs`: Sample rate (44100 Hz)
- `bits`: Bits por sample (16)
- `peak_dbfs`: Nivel de pico en dBFS
- `source`: Método de generación
- `soundfont`: Archivo soundfont utilizado

## ⚙️ Configuración

Puedes modificar los parámetros en `GenerarNotas.py`:

```python
# Configuración de audio
FS = 44100                    # Sample rate
BITS = 16                     # Bits por sample
LEAD_SIL = 0.25              # Silencio inicial (segundos)
TAIL_SIL = 0.25              # Silencio final (segundos)

# Variaciones a generar
VELOCITIES = [30, 60, 90, 110]              # Velocidades MIDI
ARTICULATIONS = {"staccato": 0.30, "sustain": 1.50}  # Duraciones
PEDALS = {"noped": 0, "ped": 127}           # Configuraciones de pedal

# Rango de notas
MIDI_MIN, MIDI_MAX = 21, 108                # A0 a C8 (piano completo)
```

## 🧪 Scripts de Prueba

El proyecto incluye scripts de prueba para validar el funcionamiento:

- `test_notas.py` - Prueba con 2 notas (16 archivos)
- `test_1_octave.py` - Prueba con 1 octava (208 archivos)
- `test_improved_duration.py` - Prueba con duraciones mejoradas (80 archivos)

## 📈 Rendimiento

**Tiempo estimado:** 15-30 minutos para generar los 704 archivos completos

**Progreso en tiempo real:**
- Contador de archivos procesados
- Velocidad de generación (archivos/segundo)
- Tiempo estimado de finalización (ETA)

**Ejemplo de salida:**
```
Generando 704 archivos de audio del piano completo...
Rango: A0 a C8 (88 notas)
[50/704] C2_v90_sustain_ped_01 - 12.3 files/sec - ETA: 8.7min
[100/704] D3_v30_staccato_noped_01 - 11.8 files/sec - ETA: 7.2min
...
🎉 ¡Completado!
📁 704 archivos generados en: data/piano_A
⏱️ Tiempo total: 18.3 minutos
⚡ Velocidad promedio: 10.4 archivos/segundo
```

## 🔧 Dependencias

### Sistema
- **macOS/Linux**: FluidSynth
  ```bash
  # macOS
  brew install fluidsynth
  
  # Ubuntu/Debian
  sudo apt-get install fluidsynth
  ```

### Python
Ver `requeriments.txt`:
```
numpy
pandas
pretty_midi
soundfile
```

## 🎼 Uso del Dataset

Este dataset es ideal para:

- **Machine Learning**: Entrenamiento de modelos de audio
- **Análisis de audio**: Estudios de espectrogramas y características
- **Síntesis de audio**: Referencia para modelos generativos
- **Educación musical**: Ejemplos de todas las notas del piano
- **Desarrollo de aplicaciones**: Samples para apps musicales

## 📄 Licencia

Este proyecto está bajo licencia MIT. El SoundFont utilizado puede tener su propia licencia, consulta [freepats.zenvoid.org](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Si encuentras algún problema:

1. Verifica que FluidSynth esté instalado: `fluidsynth --version`
2. Confirma que el archivo `piano.sf2` esté en el directorio correcto
3. Revisa que todas las dependencias Python estén instaladas
4. Abre un issue en GitHub con detalles del error

---

## 🎵 ¡Disfruta creando tu dataset de piano! 🎹
