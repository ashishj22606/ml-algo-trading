import sys
from dotenv import load_dotenv
from src.data_processing.data_generation import data_generation

load_dotenv()

def main():
    data_generation()
    
if __name__ == "__main__":
    main()