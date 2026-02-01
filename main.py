# main.py
from src.dataset_generation import create_dataset
from src.generate_data_yaml import generate_data_yaml
from src.training import train_model

if __name__ == "__main__":
    create_dataset()
    generate_data_yaml()
    train_model()
