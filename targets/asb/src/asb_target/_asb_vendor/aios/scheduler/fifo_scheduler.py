# This implements a (mostly) FIFO task queue using threads and queue, in a
# similar fashion to the round robin scheduler. However, the timeout is 1 second
# instead of 0.05 seconds.

from .base import BaseScheduler

from queue import Queue, Empty

import time

from pyopenagi.queues.llm_request_queue import LLMRequestQueue

#: Pushed by stop() to wake the blocking queue read at once. Without it, stop()
#: waits out the 1s get_message timeout, and a superred experiment tears a
#: scheduler down once PER TASK (tens of thousands of times).
_STOP = object()

class FIFOScheduler(BaseScheduler):
    def __init__(self, llm, log_mode, llm_request_queue=None):
        super().__init__(llm, log_mode)
        self.agent_process_queue = Queue()
        # superred port deviation (ASSUMPTIONS.md G.1): the queue is supplied
        # rather than taken from the class, so this scheduler serves only the
        # requests of its own runtime. Omitting it selects the shared queue,
        # which is upstream behaviour.
        self.llm_request_queue = llm_request_queue or LLMRequestQueue.default()


    def run(self):
        while self.active:
            agent_process = None
            try:
                """
                wait 1 second between each iteration at the minimum
                if there is nothing received in a second, it will raise Empty
                """
                # agent_process = self.agent_process_queue.get(block=True, timeout=1)
                agent_process = self.llm_request_queue.get_message()
                if agent_process is _STOP:  # wake-up from stop(); re-check active
                    continue
                # print("Get the request")
                agent_process.set_status("executing")
                self.logger.log(f"{agent_process.agent_name} is executing. \n", "execute")
                agent_process.set_start_time(time.time())
                self.execute_request(agent_process)
            except Empty:
                pass
            except BaseException as e:  # noqa: BLE001
                # superred port deviation (ASSUMPTIONS.md G.1): upstream let an
                # exception here kill the scheduler thread. The waiting agent
                # spins in `listen()` until a response appears, so a dead
                # scheduler hangs that agent FOREVER -- survivable when the
                # process was one run, unacceptable now that several runtimes
                # share a process. Unblock the request with a recorded failure
                # instead, and keep serving.
                self._fail_request(agent_process, e)

    def stop(self, timeout=None):
        """Stop the thread promptly, waiting at most *timeout* for it.

        superred port deviation (ASSUMPTIONS.md G.1), for two reasons. The base
        class only flips ``active`` and joins, so the thread first sleeps out
        its 1s queue timeout; a superred run tears down one scheduler per task,
        so that second is paid tens of thousands of times. Waking the queue
        makes the common case immediate.

        And the join is now bounded. If the thread is inside a request, it
        cannot return until that request does, which for a real provider is up
        to the client's read timeout. Blocking teardown for minutes would stall
        the slot; the thread is a daemon and exits on its own once the call
        returns, because ``active`` is already False.
        """
        self.active = False
        try:
            self.llm_request_queue.add_message(_STOP)
        except BaseException:  # noqa: BLE001 - fall back to the timeout
            pass
        self.thread.join(timeout)

    def _fail_request(self, agent_process, error):
        """Unblock a request whose execution raised, and record the failure."""
        try:
            from pyopenagi.utils.chat_template import Response

            failure = f"{type(error).__name__}: {error}"
            recorder = getattr(self.llm, "record_scheduler_failure", None)
            if callable(recorder):
                recorder(failure)
            if agent_process is None:  # the queue read itself failed
                return
            if agent_process.get_response() is None:
                agent_process.set_response(Response(response_message="[proxy-error]"))
            agent_process.set_end_time(time.time())
            agent_process.set_status("done")
        except BaseException:  # noqa: BLE001 - never let the loop die
            pass

    def execute_request(self, agent_process):
        self.llm.address_request(
            agent_process=agent_process
        )
