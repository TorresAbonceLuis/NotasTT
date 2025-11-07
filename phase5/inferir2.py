# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import librosa
import pretty_midi
import os
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
MODEL_PATH = "optimized_best_model.h5"
TEST_AUDIO_PATH = "3_notes_random.wav"  # Tu archivo de 3 notas
OUTPUT_MIDI_PATH = "transcripcion_corregida.mid"

# Parámetros (igual que entrenamiento)
SR = 22050
HOP_LENGTH = 512
N_MELS = 128
N_FFT = 2048
F_MIN = 25.0
F_MAX = 6000.0
LOW_MIDI = 21
HIGH_MIDI = 108

# === CARGAR MODELO ===
def load_trained_model(model_path):
    print("🎹 Cargando modelo entrenado...")
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
    
    # Extraer espectrograma Mel
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    X = S_db.T
    X = (X / 80.0) + 1.0
    
    print(f"📊 Audio procesado: {len(y)/SR:.2f}s -> {X.shape[0]} frames")
    return X.astype(np.float32), len(y)/SR

# === CREAR SECUENCIAS PARA PREDICCIÓN ===
def create_prediction_sequences(X, sequence_length=64):
    sequences = []
    n_frames = X.shape[0]
    
    for i in range(0, n_frames, sequence_length):
        if i + sequence_length <= n_frames:
            sequences.append(X[i:i + sequence_length])
        else:
            pad_length = sequence_length - (n_frames - i)
            padded = np.pad(X[i:], ((0, pad_length), (0, 0)), mode='constant')
            sequences.append(padded)
    
    return np.array(sequences)

# === PREDECIR ===
def predict_notes(model, X_sequences):
    print("🧠 Realizando predicción...")
    
    predictions = model.predict(X_sequences, verbose=1)
    onsets_pred, frames_pred = predictions
    
    # Reconstruir secuencia completa
    onsets_full = onsets_pred.reshape(-1, 88)
    frames_full = frames_pred.reshape(-1, 88)
    
    # Recortar al tamaño original
    original_frames = X_sequences.shape[0] * X_sequences.shape[1]
    onsets_full = onsets_full[:original_frames]
    frames_full = frames_full[:original_frames]
    
    return onsets_full, frames_full

# === CONVERTIR A MIDI CON UMBRALES MÁS BAJOS ===
def predictions_to_midi(onsets_pred, frames_pred, output_path, duration):
    print("🎹 Convirtiendo a MIDI...")
    
    # Crear objeto MIDI
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    hop_time = HOP_LENGTH / SR
    
    # 🔥 UMBRALES MÁS BAJOS PARA PRUEBAS INICIALES
    onset_threshold = 0.1    # Bajado de 0.3
    frame_threshold = 0.15   # Bajado de 0.4
    
    active_notes = {}
    all_notes = []
    
    print(f"🔧 Usando umbrales: onset={onset_threshold}, frame={frame_threshold}")
    
    for i, (onset_frame, frame_frame) in enumerate(zip(onsets_pred, frames_pred)):
        current_time = i * hop_time
        
        # Detectar nuevos onsets
        for pitch in range(88):
            if (onset_frame[pitch] > onset_threshold and 
                pitch not in active_notes):
                active_notes[pitch] = current_time
                print(f"🎯 Onset detectado: pitch {pitch + LOW_MIDI} en {current_time:.2f}s (confianza: {onset_frame[pitch]:.3f})")
        
        # Verificar notas que deben terminar
        notes_to_remove = []
        for pitch, start_time in active_notes.items():
            if frame_frame[pitch] < frame_threshold:
                end_time = current_time
                if end_time > start_time + 0.05:  # Mínima duración razonable
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch + LOW_MIDI,
                        start=start_time,
                        end=end_time
                    )
                    all_notes.append(note)
                    print(f"🎵 Nota terminada: pitch {pitch + LOW_MIDI} de {start_time:.2f}s a {end_time:.2f}s")
                notes_to_remove.append(pitch)
        
        # Eliminar notas terminadas
        for pitch in notes_to_remove:
            if pitch in active_notes:
                del active_notes[pitch]
    
    # Agregar notas activas restantes
    for pitch, start_time in active_notes.items():
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch + LOW_MIDI,
            start=start_time,
            end=duration
        )
        all_notes.append(note)
        print(f"🎵 Nota final: pitch {pitch + LOW_MIDI} de {start_time:.2f}s a {duration:.2f}s")
    
    # Ordenar y agregar al instrumento
    all_notes.sort(key=lambda x: x.start)
    piano.notes.extend(all_notes)
    midi.instruments.append(piano)
    
    # Guardar MIDI
    midi.write(output_path)
    print(f"💾 MIDI guardado: {output_path}")
    print(f"🎵 Total de notas detectadas: {len(all_notes)}")

# === VISUALIZAR PREDICCIONES DETALLADAS ===
def visualize_detailed_predictions(onsets_pred, frames_pred, audio_duration):
    print("📊 Creando visualización detallada...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Onsets por tiempo
    onset_activations = onsets_pred.max(axis=1)
    times = np.linspace(0, audio_duration, len(onset_activations))
    ax1.plot(times, onset_activations, 'r-', alpha=0.7, linewidth=1)
    ax1.set_title('Activación de Onsets (Máximo por frame)')
    ax1.set_ylabel('Confianza')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Umbral onset')
    
    # 2. Frames activos
    frame_activations = frames_pred.max(axis=1)
    ax2.plot(times, frame_activations, 'b-', alpha=0.7, linewidth=1)
    ax2.set_title('Activación de Frames (Máximo por frame)')
    ax2.set_ylabel('Confianza')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.15, color='blue', linestyle='--', alpha=0.5, label='Umbral frame')
    
    # 3. Piano roll de onsets
    onset_roll = (onsets_pred > 0.1).astype(float)
    ax3.imshow(onset_roll.T, aspect='auto', origin='lower', 
               extent=[0, audio_duration, LOW_MIDI, HIGH_MIDI],
               cmap='Reds', alpha=0.8)
    ax3.set_title('Onsets Detectados (umbral > 0.1)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Pitch MIDI')
    
    # 4. Piano roll de frames
    frame_roll = (frames_pred > 0.15).astype(float)
    ax4.imshow(frame_roll.T, aspect='auto', origin='lower',
               extent=[0, audio_duration, LOW_MIDI, HIGH_MIDI],
               cmap='Blues', alpha=0.8)
    ax4.set_title('Frames Activos (umbral > 0.15)')
    ax4.set_xlabel('Tiempo (s)')
    ax4.set_ylabel('Pitch MIDI')
    
    plt.tight_layout()
    plt.savefig('prediccion_detallada.png', dpi=150, bbox_inches='tight')
    print("📈 Visualización detallada guardada: prediccion_detallada.png")

# === ANALIZAR PREDICCIONES ===
def analyze_predictions(onsets_pred, frames_pred):
    print("\n🔍 ANÁLISIS DE PREDICCIONES:")
    print("=" * 40)
    
    # Estadísticas de onsets
    onset_max = onsets_pred.max()
    onset_mean = onsets_pred.mean()
    onset_above_01 = (onsets_pred > 0.1).sum()
    onset_above_02 = (onsets_pred > 0.2).sum()
    
    print(f"Onsets - Máximo: {onset_max:.3f}, Media: {onset_mean:.3f}")
    print(f"Onsets > 0.1: {onset_above_01} valores")
    print(f"Onsets > 0.2: {onset_above_02} valores")
    
    # Estadísticas de frames
    frame_max = frames_pred.max()
    frame_mean = frames_pred.mean()
    frame_above_01 = (frames_pred > 0.1).sum()
    frame_above_02 = (frames_pred > 0.2).sum()
    
    print(f"Frames - Máximo: {frame_max:.3f}, Media: {frame_mean:.3f}")
    print(f"Frames > 0.1: {frame_above_01} valores")
    print(f"Frames > 0.2: {frame_above_02} valores")
    
    # Encontrar picos más altos
    print("\n🎯 PICO MÁS ALTO EN CADA DIMENSIÓN:")
    for i in range(min(5, onsets_pred.shape[1])):  # Primeros 5 pitches
        pitch_max_onset = onsets_pred[:, i].max()
        pitch_max_frame = frames_pred[:, i].max()
        if pitch_max_onset > 0.1 or pitch_max_frame > 0.1:
            print(f"Pitch {i + LOW_MIDI}: onset_max={pitch_max_onset:.3f}, frame_max={pitch_max_frame:.3f}")

# === FUNCIÓN PRINCIPAL ===
def main():
    print("🎹 TRANSCRIPCIÓN CORREGIDA - UMBRALES BAJOS")
    print("=" * 50)
    
    try:
        # 1. Cargar modelo
        model = load_trained_model(MODEL_PATH)
        
        # 2. Verificar archivo
        if not os.path.exists(TEST_AUDIO_PATH):
            print(f"❌ No se encuentra: {TEST_AUDIO_PATH}")
            return
        
        # 3. Preprocesar audio
        X, audio_duration = preprocess_audio(TEST_AUDIO_PATH)
        
        # 4. Crear secuencias
        X_sequences = create_prediction_sequences(X)
        print(f"📈 Secuencias para predicción: {X_sequences.shape}")
        
        # 5. Predecir
        onsets_pred, frames_pred = predict_notes(model, X_sequences)
        
        # 6. Analizar predicciones
        analyze_predictions(onsets_pred, frames_pred)
        
        # 7. Convertir a MIDI (con umbrales bajos)
        predictions_to_midi(onsets_pred, frames_pred, OUTPUT_MIDI_PATH, audio_duration)
        
        # 8. Visualizar
        visualize_detailed_predictions(onsets_pred, frames_pred, audio_duration)
        
        print("\n✅ ¡PROCESO COMPLETADO!")
        print(f"🎵 Audio: {TEST_AUDIO_PATH}")
        print(f"🎹 MIDI: {OUTPUT_MIDI_PATH}")
        print(f"⏱️ Duración: {audio_duration:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()