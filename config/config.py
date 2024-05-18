from dataclasses import dataclass

@dataclass
class Config:
    img_list_path: str
    subset_size: int
    batch_size: int
    n_epochs: int
    log_every: int
    server_port: str
    save_every: int
