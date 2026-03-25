from kani import Kani

class LogTruncationKani(Kani):
    def _prune_history(self, reserve: int = 0):
        before = len(self.chat_history)
        super()._prune_history(reserve=reserve)
        after = len(self.chat_history)
        if after < before:
            print(
                f"[{self.engine.model_id}] pruned {before-after} messages; "
                f"context now {after} turns."
            )
