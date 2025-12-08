import os

AUG_RANGE = 3
DATA_DIR = "midi_data/chorales"
CACHE_FILE = "models/processed_midi.pkl"

# --- MODEL CONFIGURATIONS ---
MODEL_CONFIGS = {
    "speedster": { # Quick iteration; finding bugs; creating simple "folk" tunes.
        "SEQ_LENGTH": 64,
        "HIDDEN_SIZE": 256,
        "EMBED_DIM_PITCH": 64,
        "EMBED_DIM_DUR": 32,
        "NUM_LAYERS": 2,
        "EPOCHS": 5,
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.3,
        "TEMP_P": 1.2,
        "TEMP_D": 1.0,
        "TOP_P": 0.9,
        "MODEL_FILE": "models/speedster/model.pkl",
        "OUTPUT_FILE": "models/speedster/output.mid"
    },
    "classic": { # The "Goldilocks" zone. Best starting point for Bach Chorales.
        "SEQ_LENGTH": 96,
        "HIDDEN_SIZE": 512,
        "EMBED_DIM_PITCH": 128,
        "EMBED_DIM_DUR": 64,
        "NUM_LAYERS": 2,
        "EPOCHS": 20,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.4,
        "TEMP_P": 1.0,
        "TEMP_D": 1.0,
        "TOP_P": 0.9,
        "MODEL_FILE": "models/classic/model.pkl",
        "OUTPUT_FILE": "models/classic/output.mid"
    },
    "deep_stack": { # Learning complex voice leading (counterpoint) over raw chords.
        "SEQ_LENGTH": 128,
        "HIDDEN_SIZE": 512,
        "EMBED_DIM_PITCH": 128,
        "EMBED_DIM_DUR": 64,
        "NUM_LAYERS": 4,
        "EPOCHS": 25,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.5,
        "TEMP_P": 1.0,
        "TEMP_D": 1.1,
        "TOP_P": 0.9,
        "MODEL_FILE": "models/deep_stack/model.pkl",
        "OUTPUT_FILE": "models/deep_stack/output.mid"
    },
    "composer": { # Maintaining key signatures and phrasing over long durations.
        "SEQ_LENGTH": 256,
        "HIDDEN_SIZE": 512,
        "EMBED_DIM_PITCH": 128,
        "EMBED_DIM_DUR": 64,
        "NUM_LAYERS": 2,
        "EPOCHS": 20,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.5,
        "TEMP_P": 0.9,
        "TEMP_D": 1.0,
        "TOP_P": 0.9,
        "MODEL_FILE": "models/composer/model.pkl",
        "OUTPUT_FILE": "models/composer/output.mid"
    },
    "titan": { # High-fidelity texture. Risk of overfitting (needs high dropout).
        "SEQ_LENGTH": 128,
        "HIDDEN_SIZE": 1536,
        "EMBED_DIM_PITCH": 256,
        "EMBED_DIM_DUR": 128,
        "NUM_LAYERS": 2,
        "EPOCHS": 15,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.0005,
        "DROPOUT": 0.6,
        "TEMP_P": 1.1,
        "TEMP_D": 1.2,
        "TOP_P": 0.95,
        "MODEL_FILE": "models/titan/model.pkl",
        "OUTPUT_FILE": "models/titan/output.mid"
    }
}

def load_config(config_name):

    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Config '{config_name}' not found.")
        
    config = MODEL_CONFIGS[config_name]

    folder_path = os.path.dirname(config["MODEL_FILE"])
    if not os.path.exists(folder_path):
        print(f"Creating directory: {folder_path}")
        os.makedirs(folder_path)

    print(f"Current Model: {config_name}")
    return config
