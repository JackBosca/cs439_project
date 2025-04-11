import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Main configuration class"""
    dataset_name: str
    model_name: str
    max_length: int
    num_workers: int
    pin_memory: bool
    min_text_length: int
    tokenizer: Optional[object] = None
    vocab_size: Optional[int] = None

def load_config(config_path: str, model_name: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and return Config object
    
    Args:
        config_path: Path to YAML config file
        model_name: Optional model name override
        
    Returns:
        Config: Loaded configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override model_name if provided
    if model_name is not None:
        config_dict['model_name'] = model_name
    
    # Create config object
    config = Config(**config_dict)
    
    return config

def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Config object to save
        save_path: Path to save YAML file
    """
    config_dict = {
        'dataset_name': config.dataset_name,
        'model_name': config.model_name,
        'max_length': config.max_length,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        'min_text_length': config.min_text_length
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f)