from abc import ABC, abstractmethod
from typing import List, Any

class BaseModel(ABC):

    @abstractmethod
    def batch_inference(self, inputs: List[Any]) -> List[Any]:
        """
        Run inference on a batch of individual inputs.

        This method is called by the inference engine each time a new batch
        has been collected from multiple incoming requests.  It **must** be
        implemented by every model that wants to be served by this engine.

        Args:
            inputs: A list of *individual* request payloads, exactly as they
                were submitted to the engine via ``submit()``.  The length of
                the list is guaranteed to be between 1 and ``max_batch_size``.

        Returns:
            A list of inference outputs, **one per input item**, in the same
            order as the input list.  Every item in the output list
            corresponds to the input at the same index.

        Raises:
            Any exception raised by this method will be caught by the engine
            and forwarded to **all** requests in the batch (as a
            ``Future`` exception).

        Notes:
            * This method runs **inside a worker thread** (via
              ``asyncio.to_thread``), so it may contain blocking calls
              (e.g., GPU inference, heavy computation).
            * The engine will **never** call this method concurrently for the
              same model instance – calls are serialized per scheduler.
            * Make sure the output list has exactly ``len(inputs)`` items,
              otherwise the engine will raise an ``AssertionError``.

        Example:
            >>> class MyModel(BaseModel):
            ...     def batch_inference(self, inputs):
            ...         # Preprocess, forward pass, postprocess
            ...         return [self._process(x) for x in inputs]
        """
        pass
