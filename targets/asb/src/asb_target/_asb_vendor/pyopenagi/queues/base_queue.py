import queue
import threading

class BaseQueue:
    """A request queue, owned by one runtime.

    superred port deviation (ASSUMPTIONS.md G.1). Upstream held ``_queue`` as a
    CLASS attribute with classmethod put/get, so every agent and scheduler in a
    process shared one queue. A queued request names no model, so with two
    runtimes alive whichever scheduler popped a request answered it with ITS
    kernel's model: silent cross-talk. One queue per runtime removes that.

    ``on_unservable`` is called with any message this queue can never serve.
    It is not optional cleanup: a request has no timeout (``BaseAgent.listen``
    spins until a response appears), so a message left in a queue whose
    scheduler has stopped hangs its caller for the life of the process.
    """

    def __init__(self, on_unservable):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self.on_unservable = on_unservable

    def close(self):
        """Refuse further work and release everything already waiting."""
        with self._lock:
            self._closed = True
            pending = []
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
        for message in pending:
            self._release(message)

    def _release(self, message):
        try:
            self.on_unservable(message)
        except BaseException:  # noqa: BLE001 - releasing a caller must never raise
            pass

    def add_message(self, message):
        with self._lock:
            closed = self._closed
            if not closed:
                self._queue.put(message)
        if closed:
            self._release(message)

    def get_message(self):
        return self._queue.get(block=True, timeout=1)

    def is_empty(self):
        return self._queue.empty()
