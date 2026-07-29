import queue
import threading

class BaseQueue:
    """A request queue.

    superred port deviation (see ASSUMPTIONS.md G.1): upstream held ``_queue``
    as a CLASS attribute and exposed put/get as classmethods, so every agent
    and every scheduler in a process shared one queue. A queued request
    carries no model, so with two runtimes alive whichever scheduler popped a
    request answered it with ITS kernel's model: silent cross-talk. The queue
    is per-INSTANCE here and callers take one explicitly. :meth:`default`
    keeps a single shared queue for callers that do not supply one, so a lone
    runtime behaves exactly as upstream.
    """

    _default_lock = threading.Lock()

    def __init__(self):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        #: Called with any message this queue can never serve, so the caller
        #: waiting on it is released instead of blocking forever. Set by the
        #: owning runtime; see :meth:`close`.
        self.on_unservable = None

    @classmethod
    def default(cls):
        """The shared queue used when no queue is supplied (upstream behaviour)."""
        with BaseQueue._default_lock:
            existing = cls.__dict__.get("_default_instance")
            if existing is None:
                existing = cls()
                cls._default_instance = existing
            return existing

    def close(self):
        """Refuse further work and release everything already waiting.

        superred port deviation (ASSUMPTIONS.md G.1). A request has no timeout:
        ``BaseAgent.listen`` spins until a response appears, so any request left
        in a queue whose scheduler has stopped hangs its caller FOREVER. That
        happens whenever a run outlives its scheduler, which is exactly what a
        cancelled task does (the agent runs in a thread that cancellation cannot
        stop, and its target is torn down underneath it). Closing hands every
        such message to ``on_unservable``, so the agent unwinds with a recorded
        failure instead of wedging a thread for the life of the process.
        """
        with self._lock:
            self._closed = True
            pending = []
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
        for message in pending:
            self._discard(message)

    def _discard(self, message):
        handler = self.on_unservable
        if handler is None:
            return
        try:
            handler(message)
        except BaseException:  # noqa: BLE001 - releasing a caller must never raise
            pass

    def add_message(self, message):
        with self._lock:
            if self._closed:
                closed = True
            else:
                closed = False
                self._queue.put(message)
        if closed:
            self._discard(message)

    def get_message(self):
        return self._queue.get(block=True, timeout=1)

    def is_empty(self):
        return self._queue.empty()
