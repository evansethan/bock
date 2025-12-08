import os
import torch
import numpy as np
import pickle
from fractions import Fraction
import music21
from config import CACHE_FILE, load_config
from helpers import nucleus_sample

# --- OLD CONFIGURATION (kept in for safety, will be overwritten) ---
OUTPUT_FILE = "models/classic/output.mid"
MODEL_FILE = 'models/classic/model.pkl'
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


def generate(config_name):

    config = load_config(config_name)
    globals().update(config)

    print(f"Current Hidden Size: {HIDDEN_SIZE}") # make sure config was set
    NUM_NOTES = int(SEQ_LENGTH/2)  # Length of generated song (don't generate past context window)
    print("Num notes: ", NUM_NOTES)

    # Detect Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device.type.upper()}")

    # Load Processed Data (Vocabularies)
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Error: {CACHE_FILE} not found. Run the training script first!")
    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)
        int_to_pitch = data['int_to_pitch']
        int_to_dur = data['int_to_dur']
        # We need the original sequences to seed the generation
        input_pitches = data['pitches'] 
        input_durs = data['durs']
        pitch_to_int = data['pitch_to_int'] # Needed for sequence conversion
        dur_to_int = data['dur_to_int']

    # Load Model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Error: {MODEL_FILE} not found. Train the model first!")
    print(f"Loading model from {MODEL_FILE}...")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()

    # --- GENERATION ---
    print("Starting generation...")

    # Create a sequence of IDs to seed the generation
    all_pitch_ids = [pitch_to_int[p] for p in input_pitches if p in pitch_to_int]
    all_dur_ids = [dur_to_int[d] for d in input_durs if d in dur_to_int]

    # Start with a random slice of the training data
    start_idx = np.random.randint(0, len(all_pitch_ids) - SEQ_LENGTH - 1)
    curr_p_seq = all_pitch_ids[start_idx : start_idx + SEQ_LENGTH]
    curr_d_seq = all_dur_ids[start_idx : start_idx + SEQ_LENGTH]

    generated_pitches = []
    generated_durs = []

    for i in range(NUM_NOTES):
        t_p = torch.tensor([curr_p_seq], dtype=torch.long).to(device)
        t_d = torch.tensor([curr_d_seq], dtype=torch.long).to(device)
        
        with torch.no_grad():
            pred_p, pred_d = model(t_p, t_d)
        
        # Apply Temperature
        pred_p = pred_p / TEMP_P
        pred_d = pred_d / TEMP_D
        
        # Sample new tokens
        idx_p = nucleus_sample(pred_p, top_p=TOP_P)
        idx_d = nucleus_sample(pred_d, top_p=TOP_P)
        
        # Convert back to human-readable format
        generated_pitches.append(int_to_pitch[idx_p])
        generated_durs.append(int_to_dur[idx_d])
        
        # Update sequence for the next prediction step
        curr_p_seq.append(idx_p)
        curr_d_seq.append(idx_d)
        curr_p_seq = curr_p_seq[1:]
        curr_d_seq = curr_d_seq[1:]

    # --- RECONSTRUCT & SAVE MIDI ---
    output_stream = music21.stream.Stream()
    print("Converting tokens to MIDI...")

    for p, d in zip(generated_pitches, generated_durs):
        try:
            # Handle fractional durations
            dur_val = float(Fraction(d)) if '/' in d else float(d)
                
            if '.' in p: # Chord
                el = music21.chord.Chord(p.split('.'))
            else: # Note
                el = music21.note.Note(p)
                
            el.duration.quarterLength = dur_val
            output_stream.append(el)
        except Exception as e:
            pass # Silently skip invalid tokens

    output_stream.write('midi', fp=OUTPUT_FILE)
    print(f"Success! Generated music saved to {OUTPUT_FILE}")


def create_songs():
    # generate("speedster")
    generate("classic")
    # generate("composer")
    # generate("deep_stack")


if __name__ == "__main__":
    create_songs()