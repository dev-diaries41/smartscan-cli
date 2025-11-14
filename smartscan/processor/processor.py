import asyncio
from abc import ABC, abstractmethod
from typing import Generic
from smartscan.processor.types import Input, Output, MetricsFailure, MetricsSuccess
from smartscan.processor.processor_listener import ProcessorListener
import time
from asyncio import Semaphore


class BatchProcessor(ABC, Generic[Input, Output]):
    def __init__(self, batch_size = 10, concurrency = 4, listener: None | ProcessorListener[Input, Output] = None):
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.listener = listener

    async def run(self, items: list[Input]):
        start = time.perf_counter()
        processed_count = 0
        success_count = 0

        try:
            if(len(items) <= 0):
                print(f"No items to process")
                result = MetricsSuccess()
                if self.listener is not None:
                    self.listener.on_complete(result)
                return result
            
            if self.listener is not None:
                self.listener.on_active()
            
            batch_start = 0

            while batch_start < len(items):
                semaphore = Semaphore(self.concurrency) #place holder fo rnow will be calculated dynamically based on memory
                batch_end = batch_start + self.batch_size
                batch = items[batch_start : batch_end]
                
                async def async_task(item: Input):
                    nonlocal processed_count

                    def task(item: Input) -> Output | None:
                        try:
                            return self.on_process(item)
                        except Exception as e:
                            if self.listener is not None:
                                self.listener.on_error(e, item)
                            return None
                        finally:
                            if self.listener is not None:
                                    processed_count += 1
                                    progress = processed_count / len(items)
                                    self.listener.on_progress(progress)
                    async with semaphore:
                        return await asyncio.to_thread(task, item)
                    
                tasks = [async_task(item) for item in batch]
                batch_outputs = await asyncio.gather(*tasks)
                filtered_batch_ouptputs = [out for out in batch_outputs if out is not None]
                success_count += len(filtered_batch_ouptputs)
                await self.on_batch_complete(filtered_batch_ouptputs)
                
                batch_start += self.batch_size
            
            end = time.perf_counter()
            result = MetricsSuccess(total_processed=success_count, time_elapsed=end - start)
            if self.listener is not None:
                self.listener.on_complete(result)
            return result
        except Exception as e:
            end = time.perf_counter()
            result = MetricsFailure(time_elapsed=end - start, total_processed=success_count, error=e)
            if self.listener is not None:
                self.listener.on_fail(result)
            return result
    
    @abstractmethod
    def on_process(self, item: Input) -> Output:
        pass 

    @abstractmethod
    async def on_batch_complete(self, batch: list[Output]):
        pass 

