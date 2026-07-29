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

    @classmethod
    def default(cls):
        """The shared queue used when no queue is supplied (upstream behaviour)."""
        with BaseQueue._default_lock:
            existing = cls.__dict__.get("_default_instance")
            if existing is None:
                existing = cls()
                cls._default_instance = existing
            return existing

    def add_message(self, message):
        self._queue.put(message)

    def get_message(self):
        return self._queue.get(block=True, timeout=1)

    def is_empty(self):
        return self._queue.empty()
