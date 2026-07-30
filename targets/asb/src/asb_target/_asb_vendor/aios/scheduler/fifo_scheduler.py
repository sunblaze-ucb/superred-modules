# This implements a (mostly) FIFO task queue using threads and queue, in a
# similar fashion to the round robin scheduler. However, the timeout is 1 second
# instead of 0.05 seconds.

from .base import BaseScheduler

from queue import Queue, Empty

import time

class FIFOScheduler(BaseScheduler):
    # superred port deviation (ASSUMPTIONS.md G.1): the queue is supplied, so
    # this scheduler serves only its own runtime's requests.
    def __init__(self, llm, log_mode, llm_request_queue):
        super().__init__(llm, log_mode)
        self.agent_process_queue = Queue()
        self.llm_request_queue = llm_request_queue


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
                # print("Get the request")
                agent_process.set_status("executing")
                self.logger.log(f"{agent_process.agent_name} is executing. \n", "execute")
                agent_process.set_start_time(time.time())
                self.execute_request(agent_process)
            except Empty:
                pass
            except BaseException:  # noqa: BLE001
                # superred port deviation (ASSUMPTIONS.md G.1): upstream let an
                # exception here kill the scheduler thread, which hangs the
                # waiting agent forever (it spins in listen() until a response
                # appears). Release the request the same way a stopped runtime
                # does, and keep serving.
                if agent_process is not None:
                    self.llm_request_queue._release(agent_process)


    def execute_request(self, agent_process):
        self.llm.address_request(
            agent_process=agent_process
        )
