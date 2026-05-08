import os
import argparse
import subprocess
import sys
import json
import pandas as pd

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True, 
                        help="List of models to run: bisenet eomt hardnet rfdetr unet biseunet ducknet")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--hpo-epochs", type=int, default=30)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 0. Ensure splits exist (run a small script or just import and call)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocess.pranet_datasets import get_kvasir_pairs, get_clinicdb_pairs, get_merged_train_pairs
    get_kvasir_pairs(args.data_root)
    get_clinicdb_pairs(args.data_root)
    pairs = get_merged_train_pairs(args.data_root)
    print(f"Splits confirmed/created. Total merged training pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("CRITICAL: No training pairs found! Please check your --data-root path.")
        sys.exit(1)

    summary_tables = []

    for model_name in args.models:
        print(f"\n{'='*20} Processing Model: {model_name} {'='*20}")
        
        # 1. HPO
        hpo_dir = os.path.join(args.out_dir, "hpo")
        hparams_file = os.path.join(hpo_dir, f"best_hparams_{model_name}.json")
        if not os.path.exists(hparams_file):
            cmd_hpo = [
                sys.executable, "scripts/optuna_hpo.py",
                "--model", model_name,
                "--data-root", args.data_root,
                "--out-dir", hpo_dir,
                "--n-trials", str(args.n_trials),
                "--epochs", str(args.hpo_epochs),
                "--workers", str(args.workers)
            ]
            if args.mixed: cmd_hpo.append("--mixed")
            run_cmd(cmd_hpo)
        else:
            print(f"Using existing HParams for {model_name}")

        # 2. K-Fold Training
        kfold_dir = os.path.join(args.out_dir, "training")
        results_file = os.path.join(kfold_dir, model_name, "kfold_results.json")
        if not os.path.exists(results_file):
            cmd_train = [
                sys.executable, "scripts/kfold_train.py",
                "--model", model_name,
                "--data-root", args.data_root,
                "--hparams", hparams_file,
                "--out-dir", kfold_dir,
                "--seeds"] + [str(s) for s in args.seeds] + [
                "--folds", str(args.folds),
                "--epochs", str(args.train_epochs),
                "--workers", str(args.workers)
            ]
            if args.mixed: cmd_train.append("--mixed")
            run_cmd(cmd_train)
        else:
            print(f"Using existing training results for {model_name}")

        # 3. Evaluation
        eval_dir = os.path.join(args.out_dir, "evaluation")
        summary_file = os.path.join(eval_dir, f"eval_summary_{model_name}.csv")
        if not os.path.exists(summary_file):
            # Load best hparams to get img_size for eval
            with open(hparams_file, 'r') as f:
                hp = json.load(f)
            
            cmd_eval = [
                sys.executable, "scripts/evaluate_benchmarks.py",
                "--model", model_name,
                "--data-root", args.data_root,
                "--ckpt-dir", kfold_dir,
                "--out-dir", eval_dir,
                "--img-size", str(hp["img_size"])
            ]
            if args.mixed: cmd_eval.append("--mixed")
            run_cmd(cmd_eval)
        else:
            print(f"Using existing evaluation for {model_name}")

        # 4. Collect results for consolidated table
        model_summary = pd.read_csv(summary_file, index_index=True)
        model_summary['model'] = model_name
        summary_tables.append(model_summary)

    # Final consolidated report
    if summary_tables:
        final_df = pd.concat(summary_tables)
        final_df.to_csv(os.path.join(args.out_dir, "final_pipeline_report.csv"))
        print("\nFinal Pipeline Report saved to:", os.path.join(args.out_dir, "final_pipeline_report.csv"))

if __name__ == "__main__":
    main()
