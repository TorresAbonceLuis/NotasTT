# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import librosa
import pretty_midi
import os
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
MODEL_PATH = "optimized_best_model.h5"  # Tu modelo entrenado
TEST_AUDIO_PATH = "3_notes_random.wav"  # Cambia por tu archivo
OUTPUT_MIDI_PATH = "transcripcion_resultado.mid"

# Parámetros (deben coincidir con el entrenamiento)
SR = 22050
HOP_LENGTH = 512
N_MELS = 128
N_FFT = 2048
F_MIN = 25.0
F_MAX = 6000.0
LOW_MIDI = 21
HIGH_MIDI = 108
SEQUENCE_LENGTH = 64

# === CARGAR MODELO ===
def load_trained_model(model_path):
    print("📥 Cargando modelo entrenado...")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Modelo cargado: {model_path}")
    return model

# === PREPROCESAR AUDIO ===
def preprocess_audio(audio_path):
    print("🎵 Procesando audio...")
    
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    
    # Normalizar
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    
    # Extraer espectrograma Mel (igual que en entrenamiento)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    X = S_db.T
    X = (X / 80.0) + 1.0  # Misma normalización
    
    print(f"📊 Audio procesado: {len(y)/SR:.2f}s -> {X.shape[0]} frames")
    return X.astype(np.float32), len(y)/SR

# === CREAR SECUENCIAS PARA PREDICCIÓN ===
def create_prediction_sequences(X, sequence_length=SEQUENCE_LENGTH):
    sequences = []
    n_frames = X.shape[0]
    
    # Crear secuencias sin superposición para predicción
    for i in range(0, n_frames, sequence_length):
        if i + sequence_length <= n_frames:
            sequences.append(X[i:i + sequence_length])
        else:
            # Rellenar la última secuencia si es necesario
            pad_length = sequence_length - (n_frames - i)
            padded = np.pad(X[i:], ((0, pad_length), (0, 0)), mode='constant')
            sequences.append(padded)
    
    return np.array(sequences)

# === PREDECIR ===
def predict_notes(model, X_sequences):
    print("🧠 Realizando predicción...")
    
    # Predecir
    predictions = model.predict(X_sequences, verbose=1)
    onsets_pred, frames_pred = predictions
    
    # Reconstruir secuencia completa
    onsets_full = onsets_pred.reshape(-1, 88)
    frames_full = frames_pred.reshape(-1, 88)
    
    # Recortar al tamaño original del audio
    original_frames = X_sequences.shape[0] * X_sequences.shape[1]
    onsets_full = onsets_full[:original_frames]
    frames_full = frames_full[:original_frames]
    
    return onsets_full, frames_full

# === CONVERTIR A MIDI ===
def predictions_to_midi(onsets_pred, frames_pred, output_path, duration):
    print("🎹 Convirtiendo a MIDI...")
    
    # Crear objeto MIDI
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    hop_time = HOP_LENGTH / SR  # Tiempo por frame en segundos
    
    # Umbrales (puedes ajustarlos)
    onset_threshold = 0.3
    frame_threshold = 0.4
    
    active_notes = {}
    
    for i, (onset_frame, frame_frame) in enumerate(zip(onsets_pred, frames_pred)):
        current_time = i * hop_time
        
        # Detectar nuevos onsets (inicios de notas)
        for pitch in range(88):
            if onset_frame[pitch] > onset_threshold and pitch not in active_notes:
                # Nueva nota detectada
                active_notes[pitch] = current_time
        
        # Verificar qué notas deben terminar
        notes_to_remove = []
        for pitch, start_time in active_notes.items():
            if frame_frame[pitch] < frame_threshold:
                # La nota termina
                end_time = current_time
                if end_time > start_time:  # Asegurar duración positiva
                    note = pretty_midi.Note(
                        velocity=80,  # Velocidad media
                        pitch=pitch + LOW_MIDI,  # Convertir a pitch MIDI real
                        start=start_time,
                        end=end_time
                    )
                    piano.notes.append(note)
                notes_to_remove.append(pitch)
        
        # Remover notas terminadas
        for pitch in notes_to_remove:
            if pitch in active_notes:
                del active_notes[pitch]
    
    # Agregar cualquier nota activa restante al final
    final_time = duration
    for pitch, start_time in active_notes.items():
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch + LOW_MIDI,
            start=start_time,
            end=final_time
        )
        piano.notes.append(note)
    
    # Ordenar notas por tiempo de inicio
    piano.notes.sort(key=lambda x: x.start)
    
    # Agregar instrumento al MIDI
    midi.instruments.append(piano)
    
    # Guardar archivo MIDI
    midi.write(output_path)
    print(f"💾 MIDI guardado: {output_path}")
    print(f"🎵 Notas transcritas: {len(piano.notes)}")

# === VISUALIZAR RESULTADOS ===
def visualize_predictions(onsets_pred, frames_pred, audio_duration):
    # Crear visualización simple
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Onsets
    onset_sum = onsets_pred.sum(axis=1)
    ax1.plot(np.linspace(0, audio_duration, len(onset_sum)), onset_sum)
    ax1.set_title('Detección de Inicios de Notas (Onsets)')
    ax1.set_ylabel('Activación')
    ax1.grid(True)
    
    # Frames activos
    frame_sum = frames_pred.sum(axis=1)
    ax2.plot(np.linspace(0, audio_duration, len(frame_sum)), frame_sum)
    ax2.set_title('Notas Activas (Frames)')
    ax2.set_ylabel('Número de Notas')
    ax2.set_xlabel('Tiempo (segundos)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediccion_visualizacion.png', dpi=150, bbox_inches='tight')
    print("📊 Visualización guardada: prediccion_visualizacion.png")

# === FUNCIÓN PRINCIPAL ===
def main():
    print("🎹 TRANSCRIPCIÓN DE PIANO - PRUEBA DEL MODELO")
    print("=" * 50)
    
    try:
        # 1. Cargar modelo
        model = load_trained_model(MODEL_PATH)
        
        # 2. Verificar que existe el audio de prueba
        if not os.path.exists(TEST_AUDIO_PATH):
            print(f"❌ No se encuentra el archivo: {TEST_AUDIO_PATH}")
            print("💡 Por favor, cambia TEST_AUDIO_PATH por tu archivo WAV")
            return
        
        # 3. Preprocesar audio
        X, audio_duration = preprocess_audio(TEST_AUDIO_PATH)
        
        # 4. Crear secuencias para predicción
        X_sequences = create_prediction_sequences(X)
        print(f"📈 Secuencias para predicción: {X_sequences.shape}")
        
        # 5. Predecir
        onsets_pred, frames_pred = predict_notes(model, X_sequences)
        
        # 6. Convertir a MIDI
        predictions_to_midi(onsets_pred, frames_pred, OUTPUT_MIDI_PATH, audio_duration)
        
        # 7. Visualizar
        visualize_predictions(onsets_pred, frames_pred, audio_duration)
        
        print("\n✅ ¡TRANSCRIPCIÓN COMPLETADA!")
        print(f"🎵 Audio original: {TEST_AUDIO_PATH}")
        print(f"🎹 MIDI generado: {OUTPUT_MIDI_PATH}")
        print(f"⏱️ Duración: {audio_duration:.2f} segundos")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()