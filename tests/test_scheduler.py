import pytest
import asyncio
import time
from asyncio import Future
from src.engine.scheduler import BatchScheduler


# UNIT TESTS FOR _collect_batch
# Ensures batch collection logic, timeout handling, and queue size control work correctly

@pytest.mark.asyncio
async def test_submit_return_future(scheduler):
    """
    Check if submit function pushes request to queue and returns an incomplete Future object.
    """
    future = await scheduler.submit(42)
    assert isinstance(future, Future)
    assert not future.done()
    assert scheduler._queue.qsize() == 1


@pytest.mark.asyncio
async def test_collect_batch_immediate(scheduler):
    """
    When enough requests are submitted to reach max_batch_size, 
    the batch should be collected and returned immediately (without waiting for timeout).
    """
    # default max_batch_size in fixture is 4
    for i in range(scheduler.max_batch_size):
        await scheduler.submit(i)

    start_time = time.monotonic()
    batch = await scheduler._collect_batch()
    end_time = time.monotonic()

    assert len(batch) == scheduler.max_batch_size
    assert [req.input_data for req in batch] == list(range(scheduler.max_batch_size))
    # Must return immediately (under 0.1s)
    assert (end_time - start_time) < 0.1


@pytest.mark.asyncio
async def test_collect_batch_waits_for_first(scheduler):
    """
    The _collect_batch function will block (sleep) and wait indefinitely (or until an external timeout)
    if the queue is empty (no requests incoming).
    """
    task = asyncio.create_task(scheduler._collect_batch())
    with pytest.raises(asyncio.TimeoutError):
        # Shield the task to prevent it from being cancelled when wait_for times out after 0.1s
        await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
    
    assert not task.done()

    # Push requests to unblock it (push up to max_batch_size to avoid its own internal timeout)
    for i in range(scheduler.max_batch_size):
        await scheduler.submit(i)
    
    batch = await asyncio.wait_for(task, timeout=0.1)
    assert len(batch) == scheduler.max_batch_size


@pytest.mark.asyncio
async def test_collect_batch_respects_max_size(scheduler):
    """
    Even if there are many requests queued up, each batch collected must never exceed max_batch_size.
    """
    # Push 6 items while max_batch_size is 4
    for i in range(scheduler.max_batch_size + 2):
        await scheduler.submit(i)

    # Collect the first batch, expecting exactly 4 items without over-collecting
    batch1 = await scheduler._collect_batch()
    assert len(batch1) == scheduler.max_batch_size
    assert [req.input_data for req in batch1] == list(range(scheduler.max_batch_size))

    # Verify that 2 unprocessed items remain in the queue
    assert scheduler._queue.qsize() == 2


@pytest.mark.asyncio
async def test_collect_batch_respects_timeout(scheduler):
    """
    When max_batch_size isn't reached but max_delay_ms timer expires, 
    the scheduler must close the batch and return current requests instead of waiting indefinitely.
    """
    await scheduler.submit(10)
    await scheduler.submit(20)

    start_time = time.monotonic()
    batch = await scheduler._collect_batch()
    end_time = time.monotonic()

    assert len(batch) == 2
    assert [req.input_data for req in batch] == [10, 20]
    
    # Confirm that the scheduler waited for the specified duration
    elapsed = end_time - start_time
    assert elapsed >= (scheduler.max_delay_ms / 1000.0) - 0.1


@pytest.mark.asyncio
async def test_collect_batch_returns_partial_after_timeout(scheduler):
    """
    Simulate a low-traffic scenario where only a single request arrives. 
    After the timeout, the scheduler must still process it regardless.
    """
    await scheduler.submit(5)
    
    start_time = time.monotonic()
    batch = await scheduler._collect_batch()
    end_time = time.monotonic()
    
    assert len(batch) == 1
    assert [req.input_data for req in batch] == [5]
    
    elapsed = end_time - start_time
    assert elapsed >= (scheduler.max_delay_ms / 1000.0) - 0.1


# INTEGRATION TESTS FOR run() LOOP
# Ensures infinite loop logic, model inference, and exception handling work correctly

@pytest.mark.asyncio
async def test_scheduler_run_processes_batch(scheduler):
    """
    Check if the run() loop correctly connects to the model, 
    executes batch_inference, and returns successful results to client futures.
    """
    # Start the scheduler loop in the background
    task = asyncio.create_task(scheduler.run())

    try:
        fut1 = await scheduler.submit(10)
        fut2 = await scheduler.submit(20)

        # Wait for results from the model. Your DummyModel returns input * 2.
        res1 = await fut1
        res2 = await fut2

        assert res1 == 20
        assert res2 == 40
    finally:
        # Remember to cancel the background task during cleanup
        task.cancel()


@pytest.mark.asyncio
async def test_scheduler_run_handles_model_exceptions():
    """
    System protection: If the model (due to poor implementation) raises an exception, 
    the scheduler must not crash; it should catch the error and report it to the user via future.
    """
    class BadModel:
        def batch_inference(self, batch):
            raise ValueError("CUDA out of memory error!")

    bad_scheduler = BatchScheduler(model=BadModel(), max_batch_size=2, max_delay_ms=100)
    task = asyncio.create_task(bad_scheduler.run())

    try:
        fut = await bad_scheduler.submit(10)
        
        # Expect the model error to surface at the API level
        with pytest.raises(ValueError, match="CUDA out of memory error!"):
            await fut
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_scheduler_run_handles_mismatched_length():
    """
    System protection: If 2 items are submitted but the model predicts only 1 result, 
    the results cannot be mapped to users. The scheduler must raise a mismatched error.
    """
    class MismatchedModel:
        def batch_inference(self, batch):
            # Model error: always returns 1 result regardless of input size
            return ["wrong output"]

    bad_scheduler = BatchScheduler(model=MismatchedModel(), max_batch_size=4, max_delay_ms=100)
    task = asyncio.create_task(bad_scheduler.run())

    try:
        fut1 = await bad_scheduler.submit("req1")
        fut2 = await bad_scheduler.submit("req2")
        
        # Catch the AssertionError explicitly raised by the scheduler
        with pytest.raises(AssertionError, match="mismatched results length"):
            await fut1
            
        with pytest.raises(AssertionError, match="mismatched results length"):
            await fut2
    finally:
        task.cancel()