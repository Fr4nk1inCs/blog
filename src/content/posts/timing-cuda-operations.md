---
title: 'Transparent Profiling for PyTorch CUDA Graphs'
published: 2025-12-02
draft: false
tags: ['cuda', 'torch', 'python']
toc: true
description: 'A drop-in profiler that works seamlessly with PyTorch CUDA Graphs.'
---

:::note[disclaimer]
This post is revised by Google Gemini.
:::

It is common practice to profile PyTorch code using CUDA events to measure the execution time of various GPU operations. By recording a start and end event, we can calculate the precise elapsed time on the GPU.

However, as optimizations move toward CUDA Graphs to reduce launch overhead, this standard profiling method breaks down. This post explores why that happens, how to fix it, and how to implement a transparent, "drop-in" profiler that works seamlessly with graph capture.

## Profiling via CUDA Events

`torch.cuda.Event`s record timestamps on the GPU timeline. By placing events before and after a section of code, we measure the elapsed time without stalling the CPU (until we explicitly ask for the result).

A standard Timer implementation usually looks like this:

```python title="simple timer" {12-16}
from contextlib import contextmanager

import torch
from torch.cuda import Event


class Timer:
    def __init__(self):
        self._start_event: Event = Event(enable_timing=True)
        self._end_event: Event = Event(enable_timing=True)

    @contextmanager
    def __call__(self):
        self._start_event.record()
        yield
        self._end_event.record()

    def elapsed_time(self) -> float:
        torch.cuda.synchronize()
        return self._start_event.elapsed_time(self._end_event)
```

Using this context manager is straightforward in eager execution mode:

```python title="using the timer" collapse={13-22}
timer = Timer()
a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")

for i in range(10):
    with timer():
        _ = a + b
    elapsed = timer.elapsed_time()

    print(f"[Iter {i}] Elapsed time: {elapsed:.2f} ms")

# Output:
# [Iter 0] Elapsed time: 12.18 ms
# [Iter 1] Elapsed time: 0.05 ms
# [Iter 2] Elapsed time: 0.03 ms
# [Iter 3] Elapsed time: 0.02 ms
# [Iter 4] Elapsed time: 0.02 ms
# [Iter 5] Elapsed time: 0.02 ms
# [Iter 6] Elapsed time: 0.02 ms
# [Iter 7] Elapsed time: 0.02 ms
# [Iter 8] Elapsed time: 0.02 ms
# [Iter 9] Elapsed time: 0.02 ms
```

## The Conflict with CUDA Graphs

CUDA Graphs capture a sequence of kernel launches and replays them as a single unit. The capture process is strict: it records GPU commands but does not execute them immediately.

### Attempt 1

If we attempt to use our `Timer` inside a `torch.cuda.graph` context, it fails immediately:

```python title="using timer with cuda graphs" showLineNumbers {10}
timer = Timer()
static_a = torch.empty(1000, 1000, device="cuda")
static_b = torch.empty(1000, 1000, device="cuda")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for i in range(10):
        with timer():
            _ = static_a + static_b
        elapsed = timer.elapsed_time()
        print(f"[Iter {i}] Elapsed time: {elapsed:.2f} ms")

static_a.copy_(torch.randn(1000, 1000, device="cuda"))
static_b.copy_(torch.randn(1000, 1000, device="cuda"))
graph.replay()
```

```
Traceback (most recent call last):
  File "<python-input-3>", line 10, in <module>
    elapsed = timer.elapsed_time()
  File "simple_timer.py", line 19, in elapsed_time
    torch.cuda.synchronize()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File ".venv/lib/python3.13/site-packages/torch/cuda/__init__.py", line 1083, in synchronize
    return torch._C._cuda_synchronize()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```

The error occurs because `timer.elapsed_time()` calls `torch.cuda.synchronize()`. Synchronization is a CPU-blocking operation that waits for the GPU to finish. You cannot wait for the GPU to finish a task that you are currently in the middle of capturing.

### Attempt 2: Deferred Synchronization

A logical next step is to move the synchronization outside the capture block. We record the events during capture, but check the time after replay.

```python showLineNumbers {15} title="trying to measure time after capture"
timer = Timer()
static_a = torch.empty(1000, 1000, device="cuda")
static_b = torch.empty(1000, 1000, device="cuda")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for i in range(10):
        with timer():
            _ = static_a + static_b

static_a.copy_(torch.randn(1000, 1000, device="cuda"))
static_b.copy_(torch.randn(1000, 1000, device="cuda"))
graph.replay()

elapsed = timer.elapsed_time()
print(f"[Iter 10] Elapsed time: {elapsed:.2f} ms")
```

```
Traceback (most recent call last):
  File "<python-input-2>", line 15, in <module>
    elapsed = timer.elapsed_time()
  File "simple_timer.py", line 20, in elapsed_time
    return self._start_event.elapsed_time(self._end_event)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File ".venv/lib/python3.13/site-packages/torch/cuda/streams.py", line 220, in elapsed_time
    return super().elapsed_time(end_event)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
torch.AcceleratorError: CUDA error: invalid argument
```

Surprisingly, this still fails with `CUDA error: invalid argument`.

Why? By default, PyTorch CUDA events are captured as internal nodes within the graph. They become part of the graph structure and are not accessible as standalone "timestamps" that the CPU can query externally after the fact.

### The Fix: External Events

To measure time inside a graph, the events must be created with the `external=True` flag (which maps to `cudaEventRecordExternal`). This tells CUDA that these events are "hooks" accessible by the host, even when recorded inside a graph structure.

We need to update our `Timer` class to use this flag and separate the `synchronize` step from the `elapsed_time` calculation.

```python ins='external=True' title="timer" ins={16-18} del={13}
class Timer:
    def __init__(self):
        self._start_event: Event = Event(enable_timing=True, external=True)
        self._end_event: Event = Event(enable_timing=True, external=True)

    @contextmanager
    def __call__(self):
        self._start_event.record()
        yield
        self._end_event.record()

    def elapsed_time(self) -> float:
        torch.cuda.synchronize()
        return self._start_event.elapsed_time(self._end_event)

    def synchronize(self):
        self._start_event.synchronize()
        self._end_event.synchronize()
```

With `external=True`, the graph records "trigger this event" commands. When we later `replay()` the graph, those events are triggered on the GPU timeline, and our CPU process can wait on them externally.

```python title="using timer with cuda graphs (fixed)"
timer = Timer()
static_a = torch.empty(1000, 1000, device="cuda")
static_b = torch.empty(1000, 1000, device="cuda")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for i in range(10):
        with timer():
            _ = static_a + static_b

static_a.copy_(torch.randn(1000, 1000, device="cuda"))
static_b.copy_(torch.randn(1000, 1000, device="cuda"))
graph.replay()

timer.synchronize()
elapsed = timer.elapsed_time()
print(f"[Iter 10] Elapsed time: {elapsed:.2f} ms")
# [Iter 10] Elapsed time: 0.02 ms
```

## Scaling Up: Timer Registry

In a real-world scenario, we have multiple timers inside a single graph (e.g., one for every layer of a generic LLM). We cannot synchronize immediately after every layer during replay, or we would kill performance.

Instead, we need a **Registry** pattern:

1. **Capture Phase**: Accumulate `Timer` objects in a list.
2. **Replay Phase**: Run the graph.
3. **Sync Phase**: Batch synchronize all timers and dump results.

```python title="timer registry"
@dataclass
class _TimingRecord[Context]:
    context: Context
    timer: Timer


class TimerRegistry[Context]:
    def __init__(self):
        self._timings: list[_TimingRecord[Context]] = []

    @contextmanager
    def time(self, context: Context):
        timer = Timer()
        timing = _TimingRecord(context=context, timer=timer)
        self._timings.append(timing)

        with timer():
            yield

    def elapsed_times(self) -> Iterator[tuple[Context, float]]:
        for timing in self._timings:
            timing.timer.synchronize()
            yield timing.context, timing.timer.elapsed_time()
```

```python title="using timer registry with cuda graphs" collapse={19-51}
registry = TimerRegistry()
static_a = torch.empty(1000, 1000, device="cuda")
static_b = torch.empty(1000, 1000, device="cuda")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for i in range(10):
        with registry.time(context=i):
            _ = static_a + static_b

for i in range(3):
    print(f"--- Replay Iteration {i} ---")
    static_a.copy_(torch.randn(1000, 1000, device="cuda"))
    static_b.copy_(torch.randn(1000, 1000, device="cuda"))
    graph.replay()
    for context, elapsed in registry.elapsed_times():
        print(f"[Iter {context}] {elapsed:.2} ms")
# Output:
# --- Replay Iteration 0 ---
# [Iter 0] 0.019 ms
# [Iter 1] 0.022 ms
# [Iter 2] 0.022 ms
# [Iter 3] 0.023 ms
# [Iter 4] 0.022 ms
# [Iter 5] 0.023 ms
# [Iter 6] 0.022 ms
# [Iter 7] 0.023 ms
# [Iter 8] 0.022 ms
# [Iter 9] 0.023 ms
# --- Replay Iteration 1 ---
# [Iter 0] 0.016 ms
# [Iter 1] 0.022 ms
# [Iter 2] 0.022 ms
# [Iter 3] 0.022 ms
# [Iter 4] 0.022 ms
# [Iter 5] 0.022 ms
# [Iter 6] 0.022 ms
# [Iter 7] 0.022 ms
# [Iter 8] 0.023 ms
# [Iter 9] 0.022 ms
# --- Replay Iteration 2 ---
# [Iter 0] 0.016 ms
# [Iter 1] 0.022 ms
# [Iter 2] 0.02 ms
# [Iter 3] 0.023 ms
# [Iter 4] 0.02 ms
# [Iter 5] 0.022 ms
# [Iter 6] 0.022 ms
# [Iter 7] 0.022 ms
# [Iter 8] 0.02 ms
# [Iter 9] 0.022 ms
```

## “Transparent” Profiling via Monkey-Patching

Integrating this registry into a large codebase (like Megatron-LM or vLLM) can be intrusive. You don't want to rewrite your training/generation loop to manually handle registries.

We can solve this by monkey-patching `torch.cuda.CUDAGraph`. We can inject logic to automatically attach a `TimerRegistry` to every new graph and automatically dump the timings whenever `graph.replay()` is called.

```python title="monkey-patching cudagraph"
class Profiler[Context]:
    singleton: Any = None

    def __init__(self, dump_fn: Callable[[Context, float], None]):
        self._graph2registry: dict[torch.cuda.CUDAGraph, TimerRegistry[Context]] = {}
        self._dump_fn = dump_fn
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._hook_torch_cuda_graph()

    def __new__(cls, dump_fn: Callable[[Context, float], None]) -> Self:
        if cls.singleton is None:
            cls.singleton = super(Profiler, cls).__new__(cls)
        return cast(Self, cls.singleton)

    def _hook_torch_cuda_graph(self):
        """Hook into CUDA graph APIs to manage TimerRegistries automatically."""

        profiler = self
        original_cg_init = torch.cuda.CUDAGraph.__init__
        original_cg_enter = torch.cuda.CUDAGraph.capture_begin
        original_cg_exit = torch.cuda.CUDAGraph.capture_end
        original_cg_replay = torch.cuda.CUDAGraph.replay

        def hooked_cg_init(self: torch.cuda.CUDAGraph):
            original_cg_init(self)
            profiler._graph2registry[self] = TimerRegistry()

        def hooked_cg_enter(
            self: torch.cuda.CUDAGraph,
            pool: _POOL_HANDLE | None = None,
            capture_error_mode: str = "global",
        ) -> None:
            profiler._current_graph = self
            profiler._graph2registry[self]._timings.clear()
            original_cg_enter(self, pool, capture_error_mode)

        def hooked_cg_exit(self: torch.cuda.CUDAGraph) -> None:
            original_cg_exit(self)
            profiler._current_graph = None

        def hooked_cg_replay(self: torch.cuda.CUDAGraph):
            original_cg_replay(self)
            registry = profiler._graph2registry.get(self, None)
            if registry is not None:
                for context, elapsed in registry.elapsed_times():
                    profiler._dump_fn(context, elapsed)

        torch.cuda.CUDAGraph.__init__ = hooked_cg_init
        torch.cuda.CUDAGraph.capture_begin = hooked_cg_enter
        torch.cuda.CUDAGraph.capture_end = hooked_cg_exit
        torch.cuda.CUDAGraph.replay = hooked_cg_replay

    @contextmanager
    def time(self, context: Context):
        """Time a code block within the current CUDA graph context."""
        if self._current_graph is not None:
            registry = self._graph2registry[self._current_graph]
            with registry.time(context=context):
                yield
        else:
            timer = Timer()
            with timer():
                yield
            self._dump_fn(context, timer.elapsed_time())
```

### Example Usage

Here is a simple scenario where we have a model with multiple layers. We want to profile each layer's operations during CUDA graph execution. The execution code is separated from the model definition.

```python title="model.py"
import torch

class Layer(torch.nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()

        self.layer_id = layer_id
        self.offset = torch.nn.Parameter(torch.randn(1000, 1000))
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.offset
        x = self.relu(x)
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(i) for i in range(5)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

```python title="execution.py"
import torch

from .model import Model


def execution():
    model = Model().cuda()
    static_input = torch.randn(1000, 1000, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ = model(static_input)

    for i in range(3):
        print(f"--- Replay Iteration {i} ---")
        static_input.copy_(torch.randn(1000, 1000, device="cuda"))
        graph.replay()


if __name__ == "__main__":
    execution()
```

Now, we can profile a model's internals without changing the execution loop. We simply initialize the `Profiler` in the model definition file.

```diff lang="python" title="model.py (with Profiler)"
+from dataclasses import dataclass
+
 import torch

+from .profiler import Profiler
+
+
+@dataclass
+class Context:
+    layer_id: int
+    op: str
+
+
+def dump_fn(context: Context, elapsed: float):
+    print(f"[Layer {context.layer_id}] {context.op} took {elapsed:.2f} ms")
+
+
+profiler = Profiler(dump_fn)
+

 class Layer(torch.nn.Module):
     def __init__(self, layer_id: int):
         self.relu = torch.nn.ReLU()

     def forward(self, x: torch.Tensor) -> torch.Tensor:
-        x = x + self.offset
-        x = self.relu(x)
+        with profiler.time(Context(self.layer_id, "add")):
+            x = x + self.offset
+        with profiler.time(Context(self.layer_id, "relu")):
+            x = self.relu(x)
         return x

 class Model(torch.nn.Module):
     def __init__(self):
         super().__init__()
         self.layers = torch.nn.ModuleList([Layer(i) for i in range(5)])

     def forward(self, x: torch.Tensor) -> torch.Tensor:
         for layer in self.layers:
             x = layer(x)
         return x
```

When you run `execution.py`, you'll see timing information for each layer's operations during the graph replay:

```text collapse={2-11,13-22,24-33}
--- Replay Iteration 0 ---
[Layer 0] add took 0.02 ms
[Layer 0] relu took 0.01 ms
[Layer 1] add took 0.02 ms
[Layer 1] relu took 0.01 ms
[Layer 2] add took 0.02 ms
[Layer 2] relu took 0.01 ms
[Layer 3] add took 0.02 ms
[Layer 3] relu took 0.01 ms
[Layer 4] add took 0.02 ms
[Layer 4] relu took 0.01 ms
--- Replay Iteration 1 ---
[Layer 0] add took 0.02 ms
[Layer 0] relu took 0.01 ms
[Layer 1] add took 0.02 ms
[Layer 1] relu took 0.01 ms
[Layer 2] add took 0.02 ms
[Layer 2] relu took 0.01 ms
[Layer 3] add took 0.02 ms
[Layer 3] relu took 0.01 ms
[Layer 4] add took 0.02 ms
[Layer 4] relu took 0.01 ms
--- Replay Iteration 2 ---
[Layer 0] add took 0.02 ms
[Layer 0] relu took 0.01 ms
[Layer 1] add took 0.02 ms
[Layer 1] relu took 0.01 ms
[Layer 2] add took 0.02 ms
[Layer 2] relu took 0.01 ms
[Layer 3] add took 0.02 ms
[Layer 3] relu took 0.01 ms
[Layer 4] add took 0.02 ms
[Layer 4] relu took 0.01 ms
```

## Conclusion

Profiling CUDA Graphs requires shifting from "synchronize immediately" to "record now, read later." By leveraging `external=True` events and monkey-patching `CUDAGraph`, we can build a profiling tool that provides detailed insights into graph execution without disrupting the structure of existing training or inference loops.

The full implementation is available at [GitHub Gist](https://gist.github.com/Fr4nk1inCs/9c0942fe1b1b4a26d20954f84b38be44).
