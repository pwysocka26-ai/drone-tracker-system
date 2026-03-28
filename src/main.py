import yaml
from core.app import run_app

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

if __name__ == "__main__":
    print("MAIN START")
    config = load_config()
    print("CONFIG LOADED")
    run_app(config)
