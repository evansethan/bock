import os
import glob
import music21
import torch
import torch.nn as nn


# --- DUAL EMBEDDING MODEL ---
class DualMusicLSTM(nn.Module):
    def __init__(self, n_pitch, n_dur, embed_pitch, embed_dur, hidden_size, num_layers, dropout):
        super(DualMusicLSTM, self).__init__()
        
        # Two Embedding Layers
        self.emb_pitch = nn.Embedding(n_pitch, embed_pitch)
        self.emb_dur = nn.Embedding(n_dur, embed_dur)
        
        # LSTM input size is the sum of both embeddings
        lstm_input_size = embed_pitch + embed_dur
        
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Two Output Heads
        self.pitch_head = nn.Linear(hidden_size, n_pitch)
        self.dur_head = nn.Linear(hidden_size, n_dur)

    def forward(self, x_p, x_d):
        # 1. Embed separately
        e_p = self.emb_pitch(x_p) # (Batch, Seq, Embed_Pitch)
        e_d = self.emb_dur(x_d)   # (Batch, Seq, Embed_Dur)
        
        # 2. Concatenate inputs
        x = torch.cat((e_p, e_d), dim=2) # (Batch, Seq, Embed_Pitch + Embed_Dur)
        
        # 3. LSTM
        lstm_out, _ = self.lstm(x)
        
        # 4. Take last step
        last_out = lstm_out[:, -1, :]
        
        # 5. Predict separately
        out_pitch = self.pitch_head(last_out)
        out_dur = self.dur_head(last_out)
        
        return out_pitch, out_dur


def parse_midi_files(dir_path, aug_range):
    pitches = []
    durations = []
    print(f"Parsing MIDI files in {dir_path}...")
    
    if not os.path.exists(dir_path):
        print(f"ERROR: Directory {dir_path} not found.")
        return [], []

    files = glob.glob(os.path.join(dir_path, "*.mid")) + \
            glob.glob(os.path.join(dir_path, "*.midi")) + \
            glob.glob(os.path.join(dir_path, "*.MID"))

    for i, file in enumerate(files):
        try:
            if i % 10 == 0: print(f"  Processing {i}/{len(files)}: {os.path.basename(file)}")
            midi = music21.converter.parse(file)

            # Augmentation Loop
            for interval in range(-aug_range, aug_range + 1): # Transpose +- input range
                if interval == 0:
                    score = midi
                else:
                    try: score = midi.transpose(interval)
                    except: continue

                try:
                    score.quantize([4], processOffsets=True, processDurations=True, inPlace=True)
                except: pass

                s_chords = score.chordify()

                for element in s_chords.recurse():
                    if isinstance(element, music21.chord.Chord):
                        if element.duration.quarterLength < 0.1: continue
                        sorted_pitches = sorted([n.nameWithOctave for n in element.pitches])
                        if len(sorted_pitches) > 5: sorted_pitches = sorted_pitches[:5]
                        
                        # save pitches and durations seperately
                        pitches.append('.'.join(sorted_pitches))
                        durations.append(str(element.duration.quarterLength))
                        
                    elif isinstance(element, music21.note.Note):
                        if element.duration.quarterLength < 0.1: continue
                        
                        # SAVE SEPARATELY
                        pitches.append(element.pitch.nameWithOctave)
                        durations.append(str(element.duration.quarterLength))
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return pitches, durations


def nucleus_sample(prediction, top_p):
    """Samples from the probability distribution using Nucleus Sampling."""
    probs = torch.nn.functional.softmax(prediction, dim=1).squeeze()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()