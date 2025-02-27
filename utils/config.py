import yaml
import logging 

def load_config (config_path : str) -> dict:
    """
    Loads the YAML configuration file.
    Parameters:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration parameters.
    """
    try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                logging.info(f"Configuration loaded from {config_path}")
                return config
            
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise

    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise