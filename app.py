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
from generate_song import generate_sequences
from config import load_config, CACHE_FILE


# --- CUSTOMIZABLES ---
NUM_NOTES = 32 # Specific to app.py only
CONFIG_NAME = "classic" # see config.py for other options

# Import config
LOGO_FILE = "logo.jpg"
config = load_config(CONFIG_NAME)

MODEL_FILE = config['MODEL_FILE']
SEQUENCE_LENGTH = config['SEQ_LENGTH']
TEMPERATURE_PITCH = config['TEMP_P']
TEMPERATURE_DUR = config['TEMP_D']
TOP_P = config['TOP_P']


# Page Configuration
st.set_page_config(
    page_title="Bachingbird",
    layout="centered"
)

# --- SETUP & CACHING ---
@st.cache_resource
def load_system():
    """
    Loads the model, data, and device into memory.
    """
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
    """
    Generates a MIDI file from the loaded model and data.
    """
    generated_pitches, generated_durs = generate_sequences(
        model, data, device, SEQUENCE_LENGTH, num_notes, 
        TEMPERATURE_PITCH, TEMPERATURE_DUR, TOP_P
    )

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


# User Interface

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