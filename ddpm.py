from ddpm.main import launch
import sys
import os

# Add the project root to the path (one level above the ddpm folder)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    launch()
