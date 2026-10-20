"""Model-local entry: python -m models.position_compressor.cli --help."""
from models.training_cli import main

if __name__ == '__main__':
    main(model='compressor')
