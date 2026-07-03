import numpy as np
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.classify.types import  ClassificationResult
from smartscan.cluster.types import  Cluster
from smartscan.embeds.types import StoredEmbedding

class FewShotClassifier(BatchProcessor[StoredEmbedding, ClassificationResult]):
    def __init__(self, 
                labelled_clusters: list[Cluster],
                listener: ProcessorListener[StoredEmbedding, ClassificationResult],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.labelled_clusters = labelled_clusters

    def on_process(self, item: StoredEmbedding) -> ClassificationResult:
        return few_shot_classify(item, self.labelled_clusters)
    
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


def few_shot_classify(item: StoredEmbedding, labelled_clusters: list[Cluster]) -> ClassificationResult:
        label = None
        best_sim = 0.0
        
        for cluster in labelled_clusters:
            try:
                similarity = np.dot(item.embedding, cluster.embedding)
            except Exception as e:
                continue
            threshold = cluster.metadata.mean_similarity - cluster.metadata.std_similarity
            if similarity > best_sim and similarity >= threshold:
                label = cluster.label
                best_sim = similarity
        return ClassificationResult(item_id=item.item_id, label=label, similarity=float(best_sim))
