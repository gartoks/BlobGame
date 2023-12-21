from dataclasses import dataclass, field
from typing import List

@dataclass
class Blob:
    x: float
    y: float
    type: int


BLOB_SIZE: int = 4*3
PACKET_SIZE: int = 46

@dataclass
class FramePacket:
    blob_count: int = 0
    current_blob_type: int = 0
    next_blob_type: int = 0
    held_blob_type: int = 0
    current_score: int = 0
    game_index: int = 0
    can_spawn_blob: bool = False
    is_game_over: bool = False
    game_mode_key: str = ""
    
    blobs: List[Blob] = field(default_factory=list)