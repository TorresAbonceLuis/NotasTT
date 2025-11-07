# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === CONFIGURACIÓN MEJORADA ===
NPZ_DIR = "Training/adaptive_features_Spectrogram"
BATCH_SIZE = 32
EPOCHS = 100  # Más épocas pero con early stopping
SEQUENCE_LENGTH = 128
INITIAL_LR = 0.0005  # Learning rate más bajo

# === CARGAR Y PREPARAR DATOS ===
def load_and_prepare_data(npz_dir, sequence_length=SEQUENCE_LENGTH):
    X_list, Y_frames_list, Y_onsets_list = [], [], []
    
    for file in os.listdir(npz_dir):
        if file.endswith(".npz"):
            data = np.load(os.path.join(npz_dir, file))
            X_list.append(data['X'])
            Y_frames_list.append(data['Y_frames'])
            Y_onsets_list.append(data['Y_onsets'])
    
    X = np.concatenate(X_list)
    Y_frames = np.concatenate(Y_frames_list)
    Y_onsets = np.concatenate(Y_onsets_list)
    
    print(f"📊 Datos originales: X{X.shape}, Y_frames{Y_frames.shape}, Y_onsets{Y_onsets.shape}")
    
    # Crear secuencias para LSTM
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(0, len(data) - seq_length, seq_length // 2):  # 50% overlap
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    # Usar todos los datos (ya que tenemos 131 canciones)
    X_seq = create_sequences(X, sequence_length)
    Y_frames_seq = create_sequences(Y_frames, sequence_length)
    Y_onsets_seq = create_sequences(Y_onsets, sequence_length)
    
    print(f"📈 Secuencias creadas: X{X_seq.shape}, Y_frames{Y_frames_seq.shape}, Y_onsets{Y_onsets_seq.shape}")
    
    return X_seq, Y_frames_seq, Y_onsets_seq

# === MODELO MEJORADO CON MÁS REGULARIZACIÓN ===
def create_regularized_model(input_shape, num_pitches=88):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Expandir dimensión para CNN
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # PRIMERA CAPA CNN - CON MÁS DROPOUT
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((1, 2))(x)  # Reducir solo frecuencia
    x = tf.keras.layers.Dropout(0.4)(x)  # Aumentado de 0.3 a 0.4
    
    # SEGUNDA CAPA CNN - CON MÁS DROPOUT
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((1, 2))(x)  # Reducir solo frecuencia
    x = tf.keras.layers.Dropout(0.4)(x)  # Aumentado de 0.3 a 0.4
    
    # Aplanar y mantener dimensión temporal
    x = tf.keras.layers.Reshape((input_shape[0], -1))(x)
    
    # LSTM BIDIRECCIONAL CON MÁS REGULARIZACIÓN
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Aumentado de 0.3 a 0.5
    
    # CAPA DENSA FINAL - REDUCIDA
    x = tf.keras.layers.Dense(128, activation='relu')(x)  # Reducida de 256 a 128
    x = tf.keras.layers.Dropout(0.4)(x)  # Aumentado de 0.2 a 0.4
    
    # DOS SALIDAS QUE MANTIENEN DIMENSIÓN TEMPORAL
    onset_output = tf.keras.layers.Dense(num_pitches, activation='sigmoid', name='onsets')(x)
    frame_output = tf.keras.layers.Dense(num_pitches, activation='sigmoid', name='frames')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[onset_output, frame_output])
    return model

# === CALLBACKS PARA PREVENIR SOBREAJUSTE ===
def get_training_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=8,  # Más paciencia para datasets grandes
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,    # Reducir LR a la mitad
            patience=5,    # Esperar 5 épocas sin mejora
            min_lr=1e-6,   # LR mínimo
            verbose=1
        ),
        ModelCheckpoint(
            'best_piano_transcriber_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

# === ENTRENAMIENTO PRINCIPAL ===
def main():
    print("🎹 INICIANDO ENTRENAMIENTO MEJORADO CON 131 CANCIONES")
    print("=" * 60)
    
    try:
        # 1. CARGAR DATOS
        print("📥 Cargando datos...")
        X_seq, Y_frames_seq, Y_onsets_seq = load_and_prepare_data(NPZ_DIR)
        
        # 2. CREAR MODELO REGULARIZADO
        print("🔄 Creando modelo con regularización mejorada...")
        model = create_regularized_model(
            input_shape=(SEQUENCE_LENGTH, 128)
        )
        
        # 3. COMPILAR CON LEARNING RATE MÁS BAJO
        optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer,
            loss={'onsets': 'binary_crossentropy', 'frames': 'binary_crossentropy'},
            metrics={'onsets': 'accuracy', 'frames': 'accuracy'}
        )
        
        # 4. MOSTRAR RESUMEN
        model.summary()
        print(f"✅ Modelo creado con {model.count_params():,} parámetros")
        
        # 5. OBTENER CALLBACKS
        callbacks = get_training_callbacks()
        
        # 6. ENTRENAR
        print("🚀 Iniciando entrenamiento con protección contra sobreajuste...")
        print("📊 Monitorizando: val_loss (Early Stopping)")
        print("💾 Guardando mejor modelo: best_piano_transcriber_model.h5")
        
        history = model.fit(
            X_seq,
            {'onsets': Y_onsets_seq, 'frames': Y_frames_seq},
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        # 7. GUARDAR MODELO FINAL
        model.save("piano_transcriber_model_final.h5")
        print("✅ Entrenamiento completado y modelo guardado!")
        
        # 8. MOSTRAR RESULTADOS FINALES
        final_val_loss = min(history.history['val_loss'])
        final_val_accuracy = max(history.history['val_frames_accuracy'])
        best_epoch = history.history['val_loss'].index(final_val_loss) + 1
        
        print("\n" + "=" * 60)
        print("🎯 RESULTADOS FINALES:")
        print(f"   Mejor época: {best_epoch}")
        print(f"   Mejor val_loss: {final_val_loss:.4f}")
        print(f"   Mejor val_accuracy: {final_val_accuracy:.4f}")
        print(f"   Épocas entrenadas: {len(history.history['loss'])}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

# === EJECUCIÓN ===
if __name__ == "__main__":
    main()