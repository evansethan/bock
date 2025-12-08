from train_model import create_models
from generate_song import create_songs

def main():
    user_input = input("Train models from scratch? (y/n): ")

    if user_input.lower() == 'y':
        create_models()
    else:
        print("Using existing models. Generating songs...")

    create_songs()

if __name__ == "__main__":
    main()