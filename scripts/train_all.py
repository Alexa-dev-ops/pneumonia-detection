import sys
import os
import tensorflow as tf

# Add the root directory to the path so src imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_ROOT, BATCH_SIZE, STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE1_LR, STAGE2_LR
from src.dataset import collect_all_images, stratified_split, XRayDataGenerator, get_class_weights
from src.model import ARCHITECTURES, build_model, compile_model, get_callbacks

def main():
    print("Initializing Data Pipeline...")
    all_paths, all_labels = collect_all_images(DATA_ROOT)
    
    if len(all_paths) == 0:
        print(f"ERROR: No images found in {DATA_ROOT}. Ensure dataset is downloaded and extracted.")
        return

    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = stratified_split(all_paths, all_labels)
    
    train_gen = XRayDataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_gen   = XRayDataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, augment=False, shuffle=False)
    
    class_weights = get_class_weights(y_train)

    print("\nStarting Unified Training Loop...")
    
    for arch in ARCHITECTURES.keys():
        print(f'\n{"="*60}')
        print(f'Training {arch}')
        print(f'{"="*60}')
        
        tf.keras.backend.clear_session()
        model, base = build_model(arch)

        # Stage 1: Frozen Backbone
        print('\nStage 1: Head only (backbone frozen)')
        compile_model(model, STAGE1_LR)
        model.fit(
            train_gen, validation_data=val_gen,
            epochs=STAGE1_EPOCHS, class_weight=class_weights,
            callbacks=get_callbacks(arch, 1)
        )

        # Stage 2: Fine-Tuning
        print('\nStage 2: Fine-tuning top 30 backbone layers')
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
            
        compile_model(model, STAGE2_LR)
        model.fit(
            train_gen, validation_data=val_gen,
            epochs=STAGE2_EPOCHS, class_weight=class_weights,
            callbacks=get_callbacks(arch, 2)
        )
        
        print(f'\n{arch} training complete.')

    print("\nAll 5 models trained successfully. Weights saved to outputs/models/.")

if __name__ == "__main__":
    main()