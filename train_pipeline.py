import os
import json
from dataset_loader import PneumoniaDatasetLoader
from cross_subject_cv import CrossSubjectValidator
from best_prediction import BestPredictionEngine

def run_pipeline():
    print("==========================================================================")
    print("  MULTIMODAL PNEUMONIA DETECTION - CHEST X-RAY vs CHEST CT SCAN")
    print("  WITH 5-FOLD CROSS-SUBJECT VALIDATION & BEST PREDICTION ENSEMBLES")
    print("==========================================================================")

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    static_dir = os.path.join(base_dir, 'static')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    loader = PneumoniaDatasetLoader(img_size=(150, 150))
    summaries = {}

    for modality in ['xray', 'ct_scan']:
        print(f"\n[Modality: {modality.upper()}] Loading dataset...")
        records = loader.load_dataset_index(data_dir, modality=modality)

        if len(records) < 20:
            print(f"    Raw dataset for {modality.upper()} not found in data_dir. Generating synthetic dataset...")
            records = loader.generate_synthetic_dataset(data_dir, modality=modality, num_subjects=60, images_per_subject=6)

        unique_subjs = set([r['subject_id'] for r in records])
        print(f"    - Total Images: {len(records)} | Unique Patient Subjects: {len(unique_subjs)}")

        # Run 5-Fold Cross-Subject CV
        validator = CrossSubjectValidator(records, modality=modality, n_splits=5, img_size=(150, 150), checkpoint_dir=checkpoint_dir)
        summary, oof_probs, oof_targets = validator.run_cross_validation(epochs=4, lr=0.0005, batch_size=16)

        # Optimize Decision Threshold
        best_thresh, thresh_metrics = BestPredictionEngine.find_optimal_threshold(oof_probs, oof_targets)
        print(f"    - Optimized Threshold (T* = {best_thresh:.2f}) -> Acc: {thresh_metrics['accuracy']*100:.2f}% | F1: {thresh_metrics['f1']:.4f} | Recall: {thresh_metrics['recall']:.4f}")

        mod_ckpt_dir = os.path.join(checkpoint_dir, modality)
        with open(os.path.join(mod_ckpt_dir, 'optimal_threshold.json'), 'w') as f:
            json.dump(thresh_metrics, f, indent=2)

        summary['optimal_threshold'] = thresh_metrics
        summaries[modality] = summary

    # Generate Comparative Analysis JSON
    comparison_analysis = BestPredictionEngine.generate_modality_comparison(summaries['xray'], summaries['ct_scan'])
    
    dashboard_data = {
        'xray': summaries['xray'],
        'ct_scan': summaries['ct_scan'],
        'comparison': comparison_analysis
    }

    with open(os.path.join(static_dir, 'cv_results.json'), 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    with open(os.path.join(static_dir, 'modality_comparison.json'), 'w') as f:
        json.dump(comparison_analysis, f, indent=2)

    print("\n==========================================================================")
    print("  COMPARATIVE PERFORMANCE METRICS SUMMARY (CHEST X-RAY vs CHEST CT SCAN)")
    print("==========================================================================")
    print(f"  Modality     | Accuracy  | Sensitivity | Specificity | F1-Score | Optimal Threshold")
    print(f"  ------------ | --------- | ----------- | ----------- | -------- | -----------------")
    print(f"  Chest X-Ray  | {summaries['xray']['mean_accuracy']*100:6.2f}%  | {summaries['xray']['mean_sensitivity']*100:9.2f}%  | {summaries['xray']['mean_specificity']*100:9.2f}%  | {summaries['xray']['mean_f1']:6.4f}   | T* = {summaries['xray']['optimal_threshold']['optimal_threshold']:.2f}")
    print(f"  Chest CT     | {summaries['ct_scan']['mean_accuracy']*100:6.2f}%  | {summaries['ct_scan']['mean_sensitivity']*100:9.2f}%  | {summaries['ct_scan']['mean_specificity']*100:9.2f}%  | {summaries['ct_scan']['mean_f1']:6.4f}   | T* = {summaries['ct_scan']['optimal_threshold']['optimal_threshold']:.2f}")
    print("==========================================================================")

if __name__ == '__main__':
    run_pipeline()
