# eval_config.py
EVAL_CONFIG = {
    "IMG_DIR": r"/media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG/images",
    "MSK_DIR": r"/media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG/masks",
    "IMG_SIZE": 256,
    "BATCH": 8,
    "SEED": 42,
    "BASE_CH": 32,
    "CKPT": "./runs_hardnet/best.pt",

    # --- edge simulation selection ---
    # None | "pi5" | "nano"
    # You can change this to pick which device to simulate by default.
    "SIMULATE_EDGE": 'None',

    # If running on a real Jetson Nano, set this True (PyTorch FP32 baseline).
    "ACTUAL_NANO": False,

    # Linux desktop–only: approximate Nano by throttling GPU SM share (requires MPS)
    # e.g., 25 means ~25% SM share. Set to None to skip.
    "MPS_SM_PERCENT": 2,

    # --- per-device profiles ---
    "PROFILES": {
        "pi5": {
            "IMG_SIZE": 256,
            "BATCH": 1,
            "MIXED": False,
            # NEW: thread control for Pi5 simulation
            "MAX_THREADS": 1,  # change to 2 or 3 to test scaling
            "INTEROP_THREADS": 1,
        },
        "nano": {
            "IMG_SIZE": 256,
            "BATCH": 1,
            "MIXED": False,
        },
    },
}
