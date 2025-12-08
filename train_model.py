import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from collections import Counter
from helpers import cache_midi_vocab, DualMusicLSTM
from config import DATA_DIR, AUG_RANGE, CACHE_FILE, load_config


# --- OLD CONFIGURATION (kept in for safety, will be overwritten) ---
OUTPUT_FILE = "old/output.mid"
MODEL_FILE = 'old/model.pkl'
SEQ_LENGTH = 128  
HIDDEN_SIZE = 1024
EMBED_DIM_PITCH = 128
EMBED_DIM_DUR = 64       
NUM_LAYERS = 2
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.5     
TEMP_P = 1.0     # Creativity for melody (higher = more creativity)
TEMP_D = 1.1       # Creativity for rhythm (higher = less repetitive rhythm)
TOP_P = 0.9   


def train(config_name):

    config = load_config(config_name)
    globals().update(config)
    print(f"Current Hidden Size: {HIDDEN_SIZE}")

    # --- LOAD DATA ---
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        all_pitches = data['pitches']
        all_durs = data['durs']
        pitch_to_int = data['pitch_to_int']
        int_to_pitch = data['int_to_pitch']
        dur_to_int = data['dur_to_int']
        int_to_dur = data['int_to_dur']
    else:
        all_pitches, all_durs, pitch_to_int, dur_to_int = cache_midi_vocab(DATA_DIR, AUG_RANGE, CACHE_FILE)

    n_pitch_vocab = len(pitch_to_int)
    n_dur_vocab = len(dur_to_int)
    print(f"Vocab Sizes -> Pitch: {n_pitch_vocab}, Duration: {n_dur_vocab}")

    # --- PREPARE SEQUENCES (DUAL INPUTS) ---
    input_pitches = []
    input_durs = []
    target_pitches = []
    target_durs = []

    for i in range(0, len(all_pitches) - SEQ_LENGTH, 1):
        seq_p = all_pitches[i:i + SEQ_LENGTH]
        seq_d = all_durs[i:i + SEQ_LENGTH]
        out_p = all_pitches[i + SEQ_LENGTH]
        out_d = all_durs[i + SEQ_LENGTH]

        if all(k in pitch_to_int for k in seq_p) and out_p in pitch_to_int:
            input_pitches.append([pitch_to_int[k] for k in seq_p])
            input_durs.append([dur_to_int[k] for k in seq_d])
            target_pitches.append(pitch_to_int[out_p])
            target_durs.append(dur_to_int[out_d])

    # Convert to Tensors
    X_p = torch.tensor(input_pitches, dtype=torch.long)
    X_d = torch.tensor(input_durs, dtype=torch.long)
    y_p = torch.tensor(target_pitches, dtype=torch.long)
    y_d = torch.tensor(target_durs, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_p, X_d, y_p, y_d)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=BATCH_SIZE, 
                                             shuffle=True, 
                                             num_workers=4, 
                                             persistent_workers=True)
                                                
    # Setup Model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Mac GPU)")
    else:
        device = torch.device("cpu")

    model = DualMusicLSTM(n_pitch_vocab, n_dur_vocab, EMBED_DIM_PITCH, EMBED_DIM_DUR, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

    # We need two loss functions (one for each head)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. TRAINING LOOP ---
    print("Starting training...")
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for b_xp, b_xd, b_yp, b_yd in dataloader:
            b_xp, b_xd = b_xp.to(device), b_xd.to(device)
            b_yp, b_yd = b_yp.to(device), b_yd.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_p, pred_d = model(b_xp, b_xd)
            
            # Calculate loss for both heads
            loss_p = criterion(pred_p, b_yp)
            loss_d = criterion(pred_d, b_yd)
            
            # Combine losses
            total_loss = loss_p + loss_d
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(dataloader):.4f}")
        if (epoch_loss < 0.6):
            print("Loss is low enough. Stopping training...")
            break

    # Save
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    print("Saved model:", config_name)


def create_models():
    train("speedster")
    # train("classic")
    # train("composer")
    # train("titan")
    # train("deep_stack")


if __name__ == "__main__":
    create_models()