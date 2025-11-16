from smartscan.processor.processor_listener import ProcessorListener

#  Placeholder for dev
class FileScannerListener(ProcessorListener[str, tuple[str, str]]):
    def on_active(self):
        print("Scanning starting...")
    def on_progress(self, progress):
        print(f"Progress: {100 * progress:.2f}%")
    def on_fail(self, result):
        print(result.error)
    def on_error(self, e, item):
        print(f"Error processing file: {item} | {e}")
