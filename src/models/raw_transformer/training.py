"""Direct Python entry point for the raw-token baseline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.raw_transformer.trainer import main

if __name__=="__main__":main("raw_transformer")
