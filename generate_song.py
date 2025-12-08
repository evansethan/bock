import os
import torch
import numpy as np
import pickle
from fractions import Fraction
import music21
from config import CACHE_FILE, load_config
from helpers import nucleus_sample, DualMusicLSTM

# Default setup - overwritten by load_config()
OUTPUT_FILE = "models/classic/output.mid"
MODEL_FILE = 'models/classic/model.pkl'
SEQ_LENGTH = 128  
HIDDEN_SIZE = 1024
EMBED_DIM_PITCH = 128
EMBED_DIM_DUR = 64       
NUM_LAYERS = 2
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.5     
TEMP_P = 1.0     # Creativity for melody (higher = more creativity)
TEMP_D = 1.1       # Creativity for rhythm (higher = less repetitive rhythm)
TOP_P = 0.9   


def generate_sequences(model, data, device, seq_length, num_notes, temp_p, temp_d, top_p):
    """
    Generates sequences of musical notes using the trained model.
    """

    int_to_pitch = data['int_to_pitch']
    int_to_dur = data['int_to_dur']
    input_pitches = data['pitches']
    input_durs = data['durs']
    pitch_to_int = data['pitch_to_int']
    dur_to_int = data['dur_to_int']
    
    # Seed generation
    all_pitch_ids = [pitch_to_int[p] for p in input_pitches if p in pitch_to_int]
    all_dur_ids = [dur_to_int[d] for d in input_durs if d in dur_to_int]
    
    start_idx = np.random.randint(0, len(all_pitch_ids) - seq_length - 1)
    curr_p_seq = all_pitch_ids[start_idx : start_idx + seq_length]
    curr_d_seq = all_dur_ids[start_idx : start_idx + seq_length]

    generated_pitches = []
    generated_durs = []

    for _ in range(num_notes):
        t_p = torch.tensor([curr_p_seq], dtype=torch.long).to(device)
        t_d = torch.tensor([curr_d_seq], dtype=torch.long).to(device)
        
        with torch.no_grad():
            pred_p, pred_d = model(t_p, t_d)
        
        pred_p /= temp_p
        pred_d /= temp_d
        
        idx_p = nucleus_sample(pred_p, top_p=top_p)
        idx_d = nucleus_sample(pred_d, top_p=top_p)
        
        generated_pitches.append(int_to_pitch[idx_p])
        generated_durs.append(int_to_dur[idx_d])
        
        curr_p_seq.append(idx_p)
        curr_d_seq.append(idx_d)
        curr_p_seq = curr_p_seq[1:]
        curr_d_seq = curr_d_seq[1:]
    
    return generated_pitches, generated_durs


def generate(config_name):
    """
    Generates a full musical piece from a specified configuration.
    """

    config = load_config(config_name)
    globals().update(config)

    print(f"Current Hidden Size: {HIDDEN_SIZE}") # make sure config was set
    num_notes = int(SEQ_LENGTH/2)  # stay inside context window
    print("Num notes: ", num_notes)

    # Detect Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device.type.upper()}")

    # Load Processed Data (Vocabularies)
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Error: {CACHE_FILE} not found. Run the training script first!")
    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)

    # Load Model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Error: {MODEL_FILE} not found. Train the model first!")
    print(f"Loading model from {MODEL_FILE}...")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()

    print("Starting generation...")
    generated_pitches, generated_durs = generate_sequences(
        model, data, device, SEQ_LENGTH, num_notes, TEMP_P, TEMP_D, TOP_P
    )

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
            pass

    output_stream.write('midi', fp=OUTPUT_FILE)
    print(f"Success! Generated music saved to {OUTPUT_FILE}")


def create_songs():
    """
    Generates songs for a predefined list of model configurations.
    """
    # generate("debug")
    # generate("classic")
    # generate("composer")
    generate("deep_stack")
    # generate("titan")


if __name__ == "__main__":
    create_songs()