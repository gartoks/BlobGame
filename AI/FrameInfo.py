import dataclasses

@dataclasses.dataclass
class Blob:
    x: float
    y: float
    type: int


BLOB_SIZE: int = 4*3

@dataclasses.dataclass
class FramePacket:
    blob_count: int = 0
    # blobs: list[Blob]
    current_blob_type: int = 0
    next_blob_type: int = 0
    held_blob_type: int = 0
    current_score: int = 0
    game_index: int = 0
    can_spawn_blob: bool = False
    is_game_over: bool = False
    game_mode_key: str = ""