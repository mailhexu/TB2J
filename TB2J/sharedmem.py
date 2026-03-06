"""
Lightweight helpers for passing NumPy arrays through shared memory across
multiprocessing workers without serialising (copying) the data via dill/pickle.

Typical usage (owner process)
------------------------------
    shm, desc = to_shm(my_array)        # copy once into shared memory
    # ... spawn workers, pass desc ...
    free_shm(shm)                       # release when done

Inside a worker
---------------
    arr = attach_shm(desc)              # zero-copy view
    value = arr[ik].copy()
    detach_shm(arr)                     # release the worker-side handle

Or, for the common "read one row and return a copy" pattern:
    row = read_shm(desc, ik)
"""

from multiprocessing import shared_memory
from typing import NamedTuple

import numpy as np


class ShmDescriptor(NamedTuple):
    """Serialisable metadata that identifies a shared memory block.

    Instances are small (three scalars + a string) and are safe to pass
    through dill/pickle into worker processes.
    """

    name: str
    shape: tuple
    dtype: np.dtype


# ---------------------------------------------------------------------------
# Owner-side helpers
# ---------------------------------------------------------------------------


def to_shm(
    arr: np.ndarray, label: str = ""
) -> tuple[shared_memory.SharedMemory, ShmDescriptor]:
    """Copy *arr* into a new shared memory block.

    Returns
    -------
    shm : SharedMemory
        The owner handle.  Keep it alive until all workers are done, then
        call :func:`free_shm` to release the OS resource.
    desc : ShmDescriptor
        Serialisable descriptor to pass to worker processes.
    """
    arr = np.asarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    desc = ShmDescriptor(name=shm.name, shape=arr.shape, dtype=arr.dtype)
    _size_mb = arr.nbytes / 1024**2
    _tag = f" ({label})" if label else ""
    print(
        f"[shm] allocated{_tag}: {shm.name}  shape={arr.shape}  dtype={arr.dtype}  {_size_mb:.2f} MB",
        flush=True,
    )
    return shm, desc


def free_shm(shm: shared_memory.SharedMemory) -> None:
    """Close and unlink *shm*, releasing the OS shared memory segment."""
    shm.close()
    shm.unlink()


# ---------------------------------------------------------------------------
# Worker-side helpers
# ---------------------------------------------------------------------------


def attach_shm(desc: ShmDescriptor) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    """Attach to an existing shared memory block and return a zero-copy view.

    Returns
    -------
    arr : np.ndarray
        A NumPy array backed by the shared memory (no data copy).
    shm : SharedMemory
        The worker-side handle.  Call :func:`detach_shm` when finished.
    """
    shm = shared_memory.SharedMemory(name=desc.name)
    arr = np.ndarray(desc.shape, dtype=desc.dtype, buffer=shm.buf)
    return arr, shm


def detach_shm(shm: shared_memory.SharedMemory) -> None:
    """Close the worker-side handle (does *not* unlink the segment)."""
    shm.close()


def read_shm(desc: ShmDescriptor, index) -> np.ndarray:
    """Attach to *desc*, copy ``arr[index]``, detach, and return the copy.

    Convenience wrapper for the very common pattern of reading a single
    slice (e.g. one k-point) from a shared array inside a worker.
    """
    arr, shm = attach_shm(desc)
    result = arr[index].copy()
    detach_shm(shm)
    return result
