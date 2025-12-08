import os
import glob
import music21
import torch
import torch.nn as nn
import pickle
from collections import Counter


# --- DUAL EMBEDDING MODEL ---
class DualMusicLSTM(nn.Module):
    """
    A dual-embedding LSTM model for music generation.
    """
    def __init__(self, n_pitch, n_dur, embed_pitch, embed_dur, hidden_size, num_layers, dropout):
        """
        Initializes the DualMusicLSTM model layers.
        """
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
        """
        Forward pass through the model.

        Args:
            x_p: Pitch input tensor.
            x_d: Duration input tensor.

        Returns:
            A tuple of pitch and duration output tensors.
        """
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
    """
    Parses MIDI files in a directory to extract pitches and durations.

    Args:
        dir_path: Path to the directory containing MIDI files.
        aug_range: The range for data augmentation by transposition.

    Returns:
        A tuple of two lists: pitches and durations.
    """
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
            if i % 50 == 0: print(f"  Processing {i}/{len(files)}: {os.path.basename(file)}")
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


def cache_midi_vocab(data_dir, aug_range, cache_file):
    """
    Parses MIDI files, creates vocabularies for pitches and durations, and caches them.

    Args:
        data_dir: Path to the directory containing MIDI files.
        aug_range: The range for data augmentation.
        cache_file: Path to the file where the cache will be saved.

    Returns:
        A tuple containing all pitches, all durations, pitch-to-int mapping, and duration-to-int mapping.
    """
    all_pitches, all_durs = parse_midi_files(data_dir, aug_range)
    
    # Create Pitch Vocabulary
    pitch_counts = Counter(all_pitches)
    vocab_pitch = sorted(list(set([p for p in all_pitches if pitch_counts[p] >= 2])))
    pitch_to_int = {p: i for i, p in enumerate(vocab_pitch)}
    int_to_pitch = {i: p for i, p in enumerate(vocab_pitch)}
    
    # Create Duration Vocabulary
    dur_counts = Counter(all_durs)
    vocab_dur = sorted(list(set(all_durs))) # Keep all durations
    dur_to_int = {d: i for i, d in enumerate(vocab_dur)}
    int_to_dur = {i: d for i, d in enumerate(vocab_dur)}
    
    print(f"Saving cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'pitches': all_pitches, 'durs': all_durs,
            'pitch_to_int': pitch_to_int, 'int_to_pitch': int_to_pitch,
            'dur_to_int': dur_to_int, 'int_to_dur': int_to_dur
        }, f)

    return all_pitches, all_durs, pitch_to_int, dur_to_int


def nucleus_sample(prediction, top_p):
    """
    Samples from the probability distribution using Nucleus Sampling.
    """
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
