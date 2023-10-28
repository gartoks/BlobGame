import dataclasses

@dataclasses.dataclass
class Blob:
    x: float
    y: float
    type: int


BLOB_SIZE: int = 4*3

@dataclasses.dataclass
class FrameInfo:
    blobs: list[Blob]
    current_blob: int
    next_blob: int
    can_drop: bool