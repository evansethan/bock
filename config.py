
import os

DATA_DIR = "midi_data/chorales"
AUG_RANGE = 3

# --- MODEL CONFIGURATIONS ---
MODEL_CONFIGS = {
    "base": {
        "SEQ_LENGTH": 128,
        "HIDDEN_SIZE": 512,
        "EMBED_DIM_PITCH": 128,
        "EMBED_DIM_DUR": 64,
        "NUM_LAYERS": 2,
        "EPOCHS": 15,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.5,
        "TEMP_P": 1.0,
        "TEMP_D": 1.0,
        "TOP_P": 0.9,
        # Paths
        "CACHE_FILE": "models/original/processed_midi.pkl",
        "MODEL_FILE": "models/original/model.pkl",
        "OUTPUT_FILE": "models/original/output.mid"
    },
    "speedster": {
        "SEQ_LENGTH": 64,
        "HIDDEN_SIZE": 256,
        "EMBED_DIM_PITCH": 64,
        "EMBED_DIM_DUR": 32,
        "NUM_LAYERS": 2,
        "EPOCHS": 30,
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.3,
        "TEMP_P": 1.2,
        "TEMP_D": 1.0,
        "TOP_P": 0.9,
        # Paths
        "CACHE_FILE": "models/speedster/processed_midi.pkl",
        "MODEL_FILE": "models/speedster/model.pkl",
        "OUTPUT_FILE": "models/speedster/output.mid"
    },
    "classic": {
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
        # Paths
        "CACHE_FILE": "models/classic/processed_midi.pkl",
        "MODEL_FILE": "models/classic/model.pkl",
        "OUTPUT_FILE": "models/classic/output.mid"
    },
    "deep_stack": {
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
        # Paths
        "CACHE_FILE": "models/deep_stack/processed_midi.pkl",
        "MODEL_FILE": "models/deep_stack/model.pkl",
        "OUTPUT_FILE": "models/deep_stack/output.mid"
    },
    "composer": {
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
        # Paths
        "CACHE_FILE": "models/composer/processed_midi.pkl",
        "MODEL_FILE": "models/composer/model.pkl",
        "OUTPUT_FILE": "models/composer/output.mid"
    },
    "titan": {
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
        # Paths
        "CACHE_FILE": "models/titan/processed_midi.pkl",
        "MODEL_FILE": "models/titan/model.pkl",
        "OUTPUT_FILE": "models/titan/output.mid"
    }
}


