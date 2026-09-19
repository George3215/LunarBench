"""Old demo launch command compatibility."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ue"))
from launch import launch

if __name__ == "__main__":
    launch(engine="both")
