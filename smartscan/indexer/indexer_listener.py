import numpy as np
from tqdm import tqdm
from smartscan.processor.processor_listener import ProcessorListener
from fastapi import WebSocket

# TODO: Websockets / SSE with desktop app
class FileIndexerListener(ProcessorListener[str, tuple[str, np.ndarray]]):
    def __init__(self):
        self.progress_bar = tqdm(total=100, desc="Indexing")

    async def on_progress(self, progress):
        self.progress_bar.n = int(progress * 100)
        self.progress_bar.refresh()
        
    async def on_fail(self, result):
        self.progress_bar.close()
        print(result.error)

    async def on_error(self, e, item):
        print(e)
    
    async def on_complete(self, result):
        self.progress_bar.close()
        print(f"Results -  Files indexed: {result.total_processed} | Time elapsed: {result.time_elapsed:.4f}s")



class FileIndexerWebSocketListener(ProcessorListener[str, tuple[str, np.ndarray]]):
    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def on_progress(self, progress):
        await self.ws.send_json({
            "type": "progress",
            "progress": progress
        })        
    
    async def on_fail(self, result):
        await self.ws.send_json({
            "type": "fail",
            "error": str(result.error)
        })

    async def on_error(self, e, item):
        await self.ws.send_json({
            "type": "error",
            "error": str(e),
            "item": item
        })

    async def on_complete(self, result):
        await self.ws.send_json({
            "type": "complete",
            "total_processed": result.total_processed,
            "time_elapsed": result.time_elapsed
        })

