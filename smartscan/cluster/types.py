from dataclasses import dataclass
from numpy import ndarray
from typing import NewType, Dict, List

__all__ = [
    "ClusterAccuracy",
    "ClusterMetadata",
    "Cluster",
    "ItemId",
    "ClusterId",
    "Assignments",
    "MergeId",
    "TargetClusters",
    "ClusterMerges",
    "ClusterResult",
]
  

ItemId = NewType("ItemId", str)
ClusterId = NewType("ClusterId", str)
Assignments = Dict[ItemId, ClusterId]

MergeId = NewType("MergeId", str)
TargetClusters = NewType("TargetClusters", List[ClusterId])
ClusterMerges = Dict[MergeId, TargetClusters]

@dataclass
class ClusterMetadata:
    prototype_size: int
    label: str
    mean_similarity: float = 0
    std_similarity: float = 0

@dataclass
class Cluster:
    UNLABELLED = "unlabelled"
    prototype_id: str
    embedding: ndarray
    metadata: ClusterMetadata

@dataclass(frozen=True)
class ClusterAccuracy:
    per_label: Dict[str, float]
    mean_accuracy: float

@dataclass(frozen=True)
class ClusterResult:
    clusters:  Dict[ClusterId, Cluster]
    assignments: Assignments
    merges: ClusterMerges




