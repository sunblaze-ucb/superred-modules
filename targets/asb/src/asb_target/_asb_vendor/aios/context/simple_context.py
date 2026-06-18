# This manages restoring and snapshotting the context.
# The file is used in the BaseLLM class and the RRScheduler class.

from aios.context.base import BaseContextManager

import os

# de-dep surgery (superred port, see ASSUMPTIONS A.3): `import torch` made lazy.
# torch is only used by the context snapshot/recover methods, which the superred
# port never calls (no GPU context restoration), so a top-level torch import would
# needlessly pull in torch (multi-second import + interpreter-exit handlers).

# import shutil

class SimpleContextManager(BaseContextManager):
    def __init__(self):
        BaseContextManager.__init__(self)

    def start(self):
        pass

    def gen_snapshot(self, pid, context):
        import torch  # lazy (see module note)

        file_path = os.path.join(self.context_dir, f"process-{pid}.pt")
        torch.save(context, file_path)

    def gen_recover(self, pid):
        import torch  # lazy (see module note)

        file_path = os.path.join(self.context_dir, f"process-{pid}.pt")
        return torch.load(file_path)

    def check_restoration(self, pid):
        return os.path.exists(os.path.join(self.context_dir, f"process-{pid}.pt"))

    def clear_restoration(self, pid):
        # print(f"Process {pid} has been deleted.")
        os.remove(os.path.join(self.context_dir, f"process-{pid}.pt"))

    def stop(self):
        pass
