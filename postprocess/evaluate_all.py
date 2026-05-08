import os
import subprocess
import sys

# Paths
# Assumes script is running from Kvasir-structured root or can resolve paths relative to it
# If running as python3 -m postprocess.evaluate_all, os.getcwd() should be Kvasir-structured
BASE_DIR = os.getcwd()
if not os.path.exists(os.path.join(BASE_DIR, "postprocess", "evaluate_all.py")):
    # Try to find base dir if run from elsewhere
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Dataset configurations
DATASETS = {
    "Kvasir": {
        "path": "/media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG",
        "script": "postprocess.evaluate"
    },
    "ldpolyp": {
        "path": "/media/iffat/DataDrive/dataset/LDPolyp",
        "script": "postprocess.evaluate_ldpolyp"
    },
    "combined": {
        "path": "/media/iffat/DataDrive/dataset/CombinedDataset",
        "script": "postprocess.evaluate"
    }
}

def main():
    print(f"Starting Batch Evaluation in {RESULTS_DIR}...")
    
    for dataset_name, info in DATASETS.items():
        dataset_results_dir = os.path.join(RESULTS_DIR, dataset_name)
        if not os.path.exists(dataset_results_dir):
            print(f"Skipping {dataset_name} (Directory not found)")
            continue
            
        print(f"\n{'='*40}")
        print(f"Processing Dataset Group: {dataset_name}")
        print(f"Data Path: {info['path']}")
        print(f"Script: {info['script']}")
        print(f"{'='*40}")
        
        # Iterate model folders
        # We look for folders containing 'best.pt'
        candidates = sorted(os.listdir(dataset_results_dir))
        
        for folder in candidates:
            model_path = os.path.join(dataset_results_dir, folder)
            if not os.path.isdir(model_path):
                continue
                
            ckpt_path = os.path.join(model_path, "best.pt")
            if not os.path.exists(ckpt_path):
                # Maybe nested or just not a run folder
                continue
            
            # Infer model name
            # Expecting "modelname_run"
            if folder.endswith("_run"):
                model_name = folder[:-4]
            else:
                model_name = folder
                
            print(f"--> Evaluating {model_name} in {folder}...")
            
            # Use venv python if available
            # Check parent directory for .venv (Kvasir/.venv)
            parent_dir = os.path.dirname(BASE_DIR)
            venv_python = os.path.join(parent_dir, ".venv", "bin", "python3")
            
            if os.path.exists(venv_python):
                python_exe = venv_python
            else:
                # Check current dir just in case
                local_venv = os.path.join(BASE_DIR, ".venv", "bin", "python3")
                if os.path.exists(local_venv):
                    python_exe = local_venv
                else:
                    python_exe = sys.executable

            cmd = [
                python_exe, "-m", info["script"],
                "--model", model_name,
                "--ckpt", ckpt_path,
                "--data-dir", info["path"],
                "--out-dir", model_path,
                "--batch", "1"
            ]
            
            # Environment setup
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{BASE_DIR}:{env.get('PYTHONPATH', '')}"
            
            # Add --mixed if you want AMP
            cmd.append("--mixed")
            
            try:
                subprocess.run(cmd, check=True, cwd=BASE_DIR, env=env)
                print(f"    [SUCCESS] Updated {os.path.join(folder, 'results.txt')}")
            except subprocess.CalledProcessError as e:
                print(f"    [FAILED] Error evaluating: {e}")
                
    print("\nBatch Evaluation Completed.")

if __name__ == "__main__":
    main()
