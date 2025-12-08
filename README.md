# Bock

Bock is a deep learning project for generating music in the style of Bach chorales. It uses a Dual Music LSTM model to learn the relationships between pitch and duration in music, and then generates new musical pieces based on what it has learned.

## Features

*   Downloads Bach chorales dataset automatically.
*   Trains a Dual Music LSTM model on the dataset.
*   Generates new music in MIDI format.
*   Provides multiple model configurations for experimentation.
*   Uses data augmentation (transposition) to increase the size of the dataset.
*   Supports nucleus sampling for more creative and controlled generation.

## Requirements

The project requires Python 3 and the following libraries:

*   torch
*   music21
*   numpy

You can install all the dependencies by running:

```bash
pip install -r requirements.txt
```

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/bock.git
    cd bock
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download the Bach chorales dataset:

    ```bash
    python chorales.py
    ```

    This will download the dataset to the `midi_data/chorales` directory.

## Usage

### Training the Models

To train the models, run the `main.py` script and enter `y` when prompted:

```bash
python main.py
Train models from scratch? (y/n): y
```

This will train the models with the configurations defined in `config.py` and save them to the `models/` directory.

### Generating Songs

To generate songs using the trained models, run the `main.py` script and enter `n` when prompted:

```bash
python main.py
Train models from scratch? (y/n): n
```

This will load the pre-trained models and generate a song for each configuration. The generated songs will be saved as MIDI files in the corresponding model directory within the `models/` directory.

## Model Configurations

Bock provides several model configurations in `config.py` that you can use to experiment with different model architectures and hyperparameters.

*   **"Debug" (Lightweight / Iteration):** Use this when you changed your data processing code and just want to see if the model works without waiting an hour. It trains very fast but might lack depth.

*   **"Classic" (Balanced):** This is likely the most stable configuration for 4-part chorales. 96 notes is roughly 24 beats (6 bars), which is a full phrase in a chorale.

*   **"Deep Stack" (Deeper Neural Network - Abstract Reasoning):** Instead of making the model wider (more hidden units), we make it deeper (more layers). Deep networks are often better at learning hierarchical rules (e.g., "If measure 1 was C major, and measure 2 was G major, then measure 4 should resolve to C").

*   **"Composer" (Long-Term Memory):** By doubling the context window to 256, the model sees way back into the past. This helps it remember the original key signature even after modulating.

*   **"Titan" (Max Capacity / High-Fidelity):** This pushes a Mac M2 chip to the limit. A hidden size of 1536 is massive for a simple LSTM. It will be able to capture extremely subtle harmonic nuances, but it is in extreme danger of overfitting. Counter that with high dropout (0.6).