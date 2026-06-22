import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_ROOT, MODEL_DIR, BATCH_SIZE
from src.dataset import collect_all_images, stratified_split, XRayDataGenerator
from src.model import ARCHITECTURES
from src.ensemble import get_predictions, optimize_ensemble_weights
from src.evaluate import evaluate_individual_models, evaluate_ensemble, plot_confusion_matrix, save_results_json

def main():
    print("Loading Data for Evaluation...")
    all_paths, all_labels = collect_all_images(DATA_ROOT)
    _, X_val_paths, X_test_paths, _, y_val, y_test = stratified_split(all_paths, all_labels)
    
    val_gen = XRayDataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, augment=False, shuffle=False)
    test_gen = XRayDataGenerator(X_test_paths, y_test, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    arch_names = list(ARCHITECTURES.keys())
    trained_models = {}
    for arch in arch_names:
        path = os.path.join(MODEL_DIR, f'{arch}_best.keras')
        if os.path.exists(path):
            trained_models[arch] = tf.keras.models.load_model(path)
        else:
            print(f"ERROR: Model {arch} not found. Run train_all.py first.")
            return

    print("Optimizing Ensemble Weights on Validation Set...")
    val_pred_matrix = np.column_stack([get_predictions(trained_models[a], val_gen) for a in arch_names])
    weights = optimize_ensemble_weights(val_pred_matrix, y_val, len(arch_names))
    print("Optimal Weights:", dict(zip(arch_names, weights)))

    print("Evaluating on Test Set...")
    test_pred_matrix = np.column_stack([get_predictions(trained_models[a], test_gen) for a in arch_names])
    
    ind_perf = evaluate_individual_models(test_pred_matrix, y_test, arch_names)
    test_ensemble_prob = test_pred_matrix @ weights
    ens_perf, test_ensemble_pred = evaluate_ensemble(test_ensemble_prob, y_test)

    print("\nEnsemble Test Performance:")
    for k, v in ens_perf.items():
        print(f"  {k.capitalize()}: {v:.4f}")

    plot_confusion_matrix(y_test, test_ensemble_pred)
    save_results_json(weights, ens_perf, ind_perf, arch_names)
    print("Evaluation complete. Results saved in outputs/.")

if __name__ == "__main__":
    main()