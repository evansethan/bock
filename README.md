# Bock

Bock is a deep learning project for generating music in the style of Bach chorales. It uses a Dual Music LSTM model to learn the relationships between pitch and duration in music, and then generates new musical pieces based on what it has learned. There is also a Web UI component (Bachingbird) for easily generating and downloading midi files.

## Responsible AI:

No copyrighted work as used for training purposes in this project. This model is *only* trained on Bach chorales, which exist in the public domain, for learning the fundamentals of harmonic progression and voice leading.

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

## Usage

### Training the Models (Only need to run once)

To train the models, run the `main.py` script and enter `y` when prompted:

```bash
python main.py
Train models from scratch? (y/n): y
```

This will get the Bach chorales corpus from music21 and save them to the `chorales/` directory. It will then train the models with the configurations defined in `config.py` and save them to the `models/` directory. It will also go ahead and generate sample output for you.

### Generating Songs (After training step)

To generate songs after loading data + training, run the `main.py` script and enter `n` when prompted:

```bash
python main.py
Train models from scratch? (y/n): n
```

This will load the pre-trained models and generate a song for each configuration. The generated songs will be saved as MIDI files in the corresponding model directory within the `models/` directory.

### Web-Based Generation (app.py)

This project also includes a web-based interface for generating music using the trained "Classic" model

1.  Make sure you have a trained model available (e.g., `models/classic/model.pkl`).
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  This will open a new tab in your web browser with the Bachingbird interface.
4.  Click the "Generate New Chorale" button to create a new musical piece. You can listen to it directly in the browser and download the MIDI file.

## Model Configurations

Bock provides several model configurations in `config.py` that you can use to experiment with different model architectures and hyperparameters.

*   **"Classic" (Balanced):** Overall the best model. Consistently produces good output, and training doesn't overload the GPU. Very stable configuration for 4-part chorales. This model is used in the Streamlit Web UI.

*   **"Debug" (Lightweight / Iteration):** Use this when you just want to see if the model works without waiting an hour. It trains very fast but lacks depth. Dont expect high quality output.

*   **"Composer" (Long-Term Memory):** By doubling the context window to 256, the model sees way back into the past. This helps it remember the original key signature even after modulating The huge context window ended up being unnecessary and just slowed town training.

*   **"Deep Stack" (Deeper Neural Network - Abstract Reasoning):** 4-layer neural network. Deep network for learning complex hierarchical rules. Takes a long time to train, and a bit overkill for Bach chorales.

*   **"Titan" (Max Capacity / High-Fidelity):** This pushes a Mac M2 chip to the limit. A hidden size of 1536 is massive for a simple LSTM. It may capture extremely subtle harmonic nuances, but it is in danger of overfitting. Takes a VERY long time to train, had to interrupt early. Maybe the output is incredible, idk. Probably not. Use the Classic model.