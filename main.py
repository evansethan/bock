from train_model import create_models
from generate_song import create_songs
from chorales import get_chorales

def main():
    """
    Trains models and runs music generation process from CLI.
    """

    get_chorales() # includes check if alredy existing

    user_input = input("Train models from scratch? (y/n): ")

    if user_input.lower() == 'y':
        create_models()
    else:
        print("Using existing models. Generating songs...")

    create_songs()

if __name__ == "__main__":
    main()