"""ConcurrentDreamWorker — async dream-episode dispatch.

S14.1 cycle-1 introduced the sync skeleton behind a Future-based
API ; C2.17 cycle-2 adds the real concurrent backend
(`threaded=True`) while preserving backward compatibility with
the two cycle-1 modes.

Modes :

* ``sync_drain=True`` (default, cycle-1) — ``submit()`` executes
  the episode immediately on the caller thread and returns an
  already-resolved Future. Pending count never exceeds capacity
  in practice.
* ``sync_drain=False, threaded=False`` (cycle-1) — ``submit()``
  enqueues without execution ; K-QUEUE capacity is enforced.
  Episodes execute sequentially on ``.drain()``, exceptions are
  aggregated and the first is re-raised after the loop completes.
* ``threaded=True`` (cycle-2) — ``submit()`` enqueues on a bounded
  ``queue.Queue`` that a single daemon-less background thread
  consumes FIFO. ``drain()`` blocks until every submitted future
  has resolved, returns their log entries in submission order,
  and re-raises the first exception observed. ``stop()`` joins
  the worker cleanly ; the class is a context manager that calls
  ``stop()`` on exit.

The threaded mode is single-consumer by design — ordering is a
DR-0 guarantee, and multi-consumer would require an order-aware
scheduler that is out of scope for C2.17.

Preserves :
  * DR-0 : every ``runtime.execute(...)`` call appends exactly
    one ``EpisodeLogEntry``. The worker thread is the sole
    producer of log entries in threaded mode.
  * K1 : per-DE budget enforced upstream by ``DreamRuntime``.
  * K-QUEUE : ``pending_count <= queue_size``. Enforced by
    ``QueueFullError`` in deferred mode and by the bounded
    ``queue.Queue`` in threaded mode (``put_nowait`` raises,
    translated to ``QueueFullError``).

Reference : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §7
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import TracebackType

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


class QueueFullError(Exception):
    """Raised when submit() exceeds queue_size.

    In deferred mode (``sync_drain=False, threaded=False``) the
    check happens synchronously against the pending list. In
    threaded mode the bounded ``queue.Queue.put_nowait`` raises
    ``queue.Full`` which is translated to this type. Both paths
    carry the K-QUEUE invariant annotation.
    """


# Sentinel placed on the threaded queue to signal worker shutdown.
_SHUTDOWN = object()


@dataclass
class ConcurrentDreamWorker:
    """Concurrent dream worker.

    ``threaded=False`` (default) preserves the cycle-1 behaviour :
    either immediate sync execution (``sync_drain=True``) or
    deferred execution on ``drain()`` (``sync_drain=False``).

    ``threaded=True`` spawns a single background worker thread on
    first ``submit()`` and resolves futures off the caller thread.
    """

    runtime: DreamRuntime
    queue_size: int = 128
    sync_drain: bool = True
    threaded: bool = False
    _pending: list[tuple[DreamEpisode, Future]] = field(
        default_factory=list, init=False, repr=False
    )
    _resolved_since_last_drain: int = field(
        default=0, init=False, repr=False
    )
    # Threaded-mode state (lazy-allocated on first submit).
    _thread_queue: queue.Queue | None = field(
        default=None, init=False, repr=False
    )
    _thread_worker: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _thread_shutdown: threading.Event | None = field(
        default=None, init=False, repr=False
    )
    _thread_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _thread_inflight: list[Future] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def pending_count(self) -> int:
        if self.threaded:
            # In threaded mode, "pending" means "submitted and not
            # yet resolved" (queued OR currently executing).
            with self._thread_lock:
                return sum(1 for f in self._thread_inflight if not f.done())
        return len(self._pending)

    # ------------------------------------------------------------
    # Public submit / drain / shutdown API
    # ------------------------------------------------------------

    def submit(self, episode: DreamEpisode) -> Future:
        """Submit a dream-episode for execution.

        Returns a Future that resolves to the ``EpisodeLogEntry``
        once the episode has been executed (immediately in
        ``sync_drain`` mode, on ``drain()`` in deferred mode, on
        the worker thread in ``threaded`` mode).

        Raises ``QueueFullError`` if the queue is at capacity in
        deferred or threaded mode.
        """
        if self.threaded:
            return self._submit_threaded(episode)

        future: Future = Future()
        if self.sync_drain:
            self._execute_one(episode, future)
            return future

        if len(self._pending) >= self.queue_size:
            raise QueueFullError(
                f"K-QUEUE: queue full: {len(self._pending)} >= "
                f"{self.queue_size} — violates queue-capacity "
                f"invariant"
            )
        self._pending.append((episode, future))
        return future

    def drain(self) -> list[EpisodeLogEntry]:
        """Execute all pending episodes and return the log entries
        in submission order.

        ``sync_drain`` mode : returns the most recent N log entries
        from ``runtime.log`` where N matches the number of futures
        resolved since the last ``drain()`` call. The drain counter
        resets to 0 after the slice is returned.

        Deferred mode (``sync_drain=False, threaded=False``) : the
        drain loop processes ALL pending futures even when individual
        episodes raise — this preserves the DR-0 accountability
        guarantee. Aggregated exceptions are surfaced after the loop
        completes by re-raising the first one ; subsequent failures
        are accessible via ``future.exception()``.

        Threaded mode : blocks until every in-flight future has
        resolved (success, failure, or cancellation), returns the
        log entries in submission order, re-raises the first
        exception observed (same semantic as deferred mode). The
        worker thread is *not* stopped by ``drain()`` — submissions
        after drain continue to be handled.
        """
        if self.threaded:
            return self._drain_threaded()

        if self.sync_drain:
            n = self._resolved_since_last_drain
            self._resolved_since_last_drain = 0
            if n == 0:
                return []
            return list(self.runtime.log[-n:])

        entries: list[EpisodeLogEntry] = []
        exceptions: list[BaseException] = []
        while self._pending:
            episode, future = self._pending.pop(0)
            self._execute_one(episode, future)
            try:
                entries.append(future.result())
            except BaseException as exc:
                exceptions.append(exc)
                # The runtime's try/finally (DR-0) guarantees a log
                # entry was appended even on handler failure ;
                # surface it in the returned list so callers see the
                # full submission-order trace.
                if (
                    self.runtime.log
                    and self.runtime.log[-1].episode_id
                    == episode.episode_id
                ):
                    entries.append(self.runtime.log[-1])
        # Deferred drains return the just-executed entries
        # directly ; the sync counter is irrelevant in this mode.
        self._resolved_since_last_drain = 0
        if exceptions:
            # Surface the first exception after draining ALL
            # remaining futures (preserves DR-0 for the rest).
            raise exceptions[0]
        return entries

    def stop(self, *, timeout: float | None = 5.0) -> None:
        """Signal the background worker thread to exit and join it.

        Idempotent : safe to call when no thread has been started
        (no-op) or after a previous ``stop()`` call. Called
        automatically by the context-manager ``__exit__``.
        """
        if not self.threaded or self._thread_worker is None:
            return
        assert self._thread_queue is not None
        assert self._thread_shutdown is not None
        self._thread_shutdown.set()
        # Sentinel unblocks the worker if it's waiting on get().
        try:
            self._thread_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            # Queue full — worker will observe shutdown event after
            # draining ; no need to force a sentinel through.
            pass
        self._thread_worker.join(timeout=timeout)
        self._thread_worker = None
        self._thread_queue = None
        self._thread_shutdown = None

    def __enter__(self) -> ConcurrentDreamWorker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _ensure_thread_started(self) -> None:
        """Lazily allocate the background thread + queue on first
        threaded ``submit()``. Guarded by ``_thread_lock`` to stay
        race-free under concurrent first submitters.
        """
        with self._thread_lock:
            if self._thread_worker is not None:
                return
            self._thread_queue = queue.Queue(maxsize=self.queue_size)
            self._thread_shutdown = threading.Event()
            self._thread_worker = threading.Thread(
                target=self._thread_loop,
                name="ConcurrentDreamWorker",
                daemon=False,
            )
            self._thread_worker.start()

    def _submit_threaded(self, episode: DreamEpisode) -> Future:
        self._ensure_thread_started()
        assert self._thread_queue is not None
        future: Future = Future()
        # Record the future before queuing so ``pending_count`` and
        # ``drain()`` see it. Using a lock keeps the list consistent
        # with queue state if the worker starts racing immediately.
        with self._thread_lock:
            self._thread_inflight.append(future)
        try:
            self._thread_queue.put_nowait((episode, future))
        except queue.Full:
            # Roll back the inflight entry on failure so
            # pending_count stays accurate.
            with self._thread_lock:
                try:
                    self._thread_inflight.remove(future)
                except ValueError:
                    pass
            raise QueueFullError(
                f"K-QUEUE: queue full: {self.queue_size} items "
                f"already queued — violates queue-capacity invariant"
            )
        return future

    def _thread_loop(self) -> None:
        """Background worker loop : pull FIFO from the queue, run
        each episode via ``_execute_one`` (DR-0 preserved), honor
        cancellation via ``set_running_or_notify_cancel``. Exits
        when the shutdown event is set AND the queue is drained.
        """
        assert self._thread_queue is not None
        assert self._thread_shutdown is not None
        while True:
            try:
                item = self._thread_queue.get(timeout=0.05)
            except queue.Empty:
                if self._thread_shutdown.is_set():
                    return
                continue

            if item is _SHUTDOWN:
                # Drain any remaining items, honoring cancellation,
                # before exiting. Preserves submission ordering of
                # whatever the user queued before stop().
                self._drain_queue_on_shutdown()
                return

            episode, future = item  # type: ignore[misc]
            # Future contract : notify cancel BEFORE running so that
            # a user cancel() before pickup is honored atomically.
            if not future.set_running_or_notify_cancel():
                # Cancelled : skip execution entirely (no log entry).
                continue
            self._execute_one(episode, future, set_running=False)

    def _drain_queue_on_shutdown(self) -> None:
        """Cancel or fail-fast any queued items left after a
        shutdown sentinel is observed. Honors cancelled futures
        silently ; remaining futures are marked with a RuntimeError
        so their ``result()`` raises rather than hanging forever.
        """
        assert self._thread_queue is not None
        while True:
            try:
                item = self._thread_queue.get_nowait()
            except queue.Empty:
                return
            if item is _SHUTDOWN:
                continue
            _, future = item  # type: ignore[misc]
            if future.cancelled() or future.done():
                continue
            future.set_exception(RuntimeError(
                "ConcurrentDreamWorker: shutdown before execution"
            ))

    def _drain_threaded(self) -> list[EpisodeLogEntry]:
        """Block until every in-flight future resolves, then return
        their entries in submission order. Re-raises the first
        exception observed (DR-0 accountability).
        """
        with self._thread_lock:
            pending = list(self._thread_inflight)

        entries: list[EpisodeLogEntry] = []
        exceptions: list[BaseException] = []
        for future in pending:
            try:
                entries.append(future.result())
            except BaseException as exc:
                # Cancelled futures expose CancelledError here ;
                # skip them because they never touched the runtime.
                from concurrent.futures import CancelledError

                if isinstance(exc, CancelledError):
                    continue
                exceptions.append(exc)

        # Consume the drained futures so subsequent submissions
        # don't re-surface them.
        with self._thread_lock:
            self._thread_inflight = [
                f for f in self._thread_inflight if f not in pending
            ]

        if exceptions:
            raise exceptions[0]
        return entries

    def _execute_one(
        self,
        episode: DreamEpisode,
        future: Future,
        *,
        set_running: bool = True,
    ) -> None:
        """Run a single episode through the runtime, resolve the
        future to the log entry.

        Defends against runtimes that append zero or multiple
        entries per ``execute()`` (DR-0 invariant : exactly one per
        call) — the future surfaces a clear ``RuntimeError`` rather
        than an opaque ``IndexError``.

        ``set_running=False`` is used by the threaded worker loop
        because it has already called
        ``future.set_running_or_notify_cancel()`` to honor the
        Python Future cancellation contract atomically with dequeue.
        """
        if set_running and not future.set_running_or_notify_cancel():
            # Future was cancelled before we could run it.
            return
        log_len_before = len(self.runtime.log)
        try:
            self.runtime.execute(episode)
        except Exception as exc:
            future.set_exception(exc)
            return
        delta = len(self.runtime.log) - log_len_before
        if delta == 1:
            entry = self.runtime.log[log_len_before]
            future.set_result(entry)
            self._resolved_since_last_drain += 1
        elif delta == 0:
            future.set_exception(RuntimeError(
                "DR-0: runtime.execute did not append any log "
                "entry for episode"
            ))
        else:
            future.set_exception(RuntimeError(
                f"DR-0: runtime.execute appended {delta} entries, "
                f"expected exactly 1"
            ))
