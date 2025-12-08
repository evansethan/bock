import streamlit as st
import torch
import pickle
import numpy as np
import music21
from fractions import Fraction
import os
import base64
from io import BytesIO
import tempfile
from helpers import DualMusicLSTM, nucleus_sample


# --- CONFIGURATION ---
MODEL_FILE = "models/classic/model.pkl"
CACHE_FILE = "models/processed_midi.pkl"
LOGO_FILE = "logo.jpg"

# Fixed Generation Parameters
SEQUENCE_LENGTH = 128
TEMPERATURE_PITCH = 1.0 
TEMPERATURE_DUR = 1.1 
TOP_P = 0.9 
NUM_NOTES = 32

# Page Configuration
st.set_page_config(
    page_title="Bachingbird",
    layout="centered"
)

# --- 1. SETUP & CACHING ---
@st.cache_resource
def load_system():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if not os.path.exists(CACHE_FILE):
        st.error(f"File not found: {CACHE_FILE}")
        return None, None, None
        
    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)

    if not os.path.exists(MODEL_FILE):
        st.error(f"File not found: {MODEL_FILE}")
        return None, None, None
        
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    model.to(device)
    model.eval()
    
    return model, data, device



def generate_midi(model, data, device, num_notes=NUM_NOTES):
    int_to_pitch = data['int_to_pitch']
    int_to_dur = data['int_to_dur']
    input_pitches = data['pitches']
    input_durs = data['durs']
    pitch_to_int = data['pitch_to_int']
    dur_to_int = data['dur_to_int']
    
    # Seed generation
    all_pitch_ids = [pitch_to_int[p] for p in input_pitches if p in pitch_to_int]
    all_dur_ids = [dur_to_int[d] for d in input_durs if d in dur_to_int]
    
    start_idx = np.random.randint(0, len(all_pitch_ids) - SEQUENCE_LENGTH - 1)
    curr_p_seq = all_pitch_ids[start_idx : start_idx + SEQUENCE_LENGTH]
    curr_d_seq = all_dur_ids[start_idx : start_idx + SEQUENCE_LENGTH]

    generated_pitches = []
    generated_durs = []

    for _ in range(num_notes):
        t_p = torch.tensor([curr_p_seq], dtype=torch.long).to(device)
        t_d = torch.tensor([curr_d_seq], dtype=torch.long).to(device)
        
        with torch.no_grad():
            pred_p, pred_d = model(t_p, t_d)
        
        pred_p /= TEMPERATURE_PITCH
        pred_d /= TEMPERATURE_DUR
        
        idx_p = nucleus_sample(pred_p, top_p=TOP_P)
        idx_d = nucleus_sample(pred_d, top_p=TOP_P)
        
        generated_pitches.append(int_to_pitch[idx_p])
        generated_durs.append(int_to_dur[idx_d])
        
        curr_p_seq.append(idx_p)
        curr_d_seq.append(idx_d)
        curr_p_seq = curr_p_seq[1:]
        curr_d_seq = curr_d_seq[1:]

    # --- RECONSTRUCTION & METADATA ---
    output_stream = music21.stream.Stream()
    
    # 1. Set Instrument to Church Organ (MIDI Program 19)
    output_stream.insert(0, music21.instrument.Organ())
    
    # 2. Set Slower Tempo
    output_stream.insert(0, music21.tempo.MetronomeMark(number=60))

    for p, d in zip(generated_pitches, generated_durs):
        try:
            dur_val = float(Fraction(d)) if '/' in d else float(d)
            if '.' in p:
                el = music21.chord.Chord(p.split('.'))
            else:
                el = music21.note.Note(p)
            el.duration.quarterLength = dur_val
            output_stream.append(el)
        except:
            pass

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        temp_path = tmp.name

    try:
        output_stream.write('midi', fp=temp_path)
        with open(temp_path, 'rb') as f:
            midi_buffer = BytesIO(f.read())
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return midi_buffer

# --- 2. UI LAYOUT ---

with st.sidebar:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width='stretch')

model, data, device = load_system()

if model is None:
    st.error("Could not load model. Please check file paths.")
else:
    if st.button("Generate New Chorale", type="primary", width='stretch'):
        with st.spinner("Composing..."):
            midi_io = generate_midi(model, data, device, NUM_NOTES)
            st.session_state['current_midi'] = midi_io
            st.rerun()

    if 'current_midi' in st.session_state:
        midi_data = st.session_state['current_midi'].getvalue()
        
        b64_midi = base64.b64encode(midi_data).decode()
        midi_url = f"data:audio/midi;base64,{b64_midi}"

        # SGM Plus soundfont is required for non-piano instruments like Organ
        html_code = f"""
        <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
        
        <div style="text-align: center; margin-top: 20px;">
            <midi-visualizer type="piano-roll" id="myVisualizer" src="{midi_url}"></midi-visualizer>
            
            <midi-player 
                src="{midi_url}" 
                sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus" 
                visualizer="#myVisualizer">
            </midi-player>
        </div>
        """
        st.components.v1.html(html_code, height=350)

        st.download_button(
            label="⬇️ Download MIDI",
            data=midi_data,
            file_name=f"bachingbird_output.mid",
            mime="audio/midi"
        )