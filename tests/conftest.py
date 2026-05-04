import pytest
import asyncio
from src.model.dummy import DummyModel
from src.engine.scheduler import BatchScheduler    

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def scheduler(dummy_model):
    return BatchScheduler(model=dummy_model, max_batch_size=4, max_delay_ms=1000)




