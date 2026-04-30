import time
from typing import List, Any
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

class DummyModel(BaseModel):
    """ 
    This is a dummy model for testing purposes
    It takes a batch of inputs and returns a batch of outputs
    """

    def __init__(self):
        super().__init__()
    
    def batch_inference(self, batch_inputs: List[Any]) -> List[Any]:
        print(f"DummyModel: Processing batch of {len(batch_inputs)} inputs")
        time.sleep(0.05)
        
        outputs = []

        for x in batch_inputs:
            if isinstance(x, (int, float)):
                outputs.append(x * 2)
            else:
                outputs.append(None)
        
        return outputs


if __name__ == "__main__":
    model = DummyModel()
    result = model.batch_inference([1, 2, 3])
    
    assert result == [2, 4, 6], f"Expected [2, 4, 6] but got {result}"
    print("DummyModel test passed!")