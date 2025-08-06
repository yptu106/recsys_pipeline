import argparse
from recbole.quick_start import run_recbole

def parse_args():
    parser = argparse.ArgumentParser(description="Run RecBole training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Run RecBole with the provided configuration
    run_recbole(config_file_list=[args.config])

if __name__ == "__main__":
    main()