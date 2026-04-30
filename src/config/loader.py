from pydantic import BaseModel, Field
import yaml
from typing import Optional
from pathlib import Path


class ModelConfig(BaseModel):
    class_path: str
    init_kwargs: dict = Field(default_factory=dict)

class SchedulerConfig(BaseModel):
    max_batch_size: int = Field(..., gt=0, description="Maximum batch size for inference")
    max_delay_ms: int = Field(..., gt=0, description="Maximum delay in milliseconds before inference")

class ServerConfig(BaseModel):
    port: int = Field(..., gt=1024, description="Port number for the server")
    host: str = Field(..., description="Host name for the server")

class Config(BaseModel):
    model: ModelConfig
    scheduler: SchedulerConfig
    server: ServerConfig

def load_config(file_path: str | Path) -> Config:
    path = Path(file_path).resolve()
    
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        
    return Config.model_validate(config_dict)


if __name__ == '__main__':
    config = load_config("./src/config/default.yaml")
    print(config)
    
