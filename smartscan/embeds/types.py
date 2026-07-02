from numpy.typing import NDArray
from typing import Literal, Optional, List
from datetime import datetime
from dataclasses import dataclass, field


__all__ = [
    "StoredEmbedding",
    "QueryResult",
    "EncoderType"
]

@dataclass
class StoredEmbedding():
    item_id: str
    created_at: datetime
    embedding: NDArray

@dataclass(frozen=True)
class QueryResult():
    ids: List[str] = field(default_factory=list)
    sims: Optional[List[float]] = None

    
EncoderType = Literal["image_encoder", "text_encoder", "face_encoder"]
