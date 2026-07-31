"""Live operational view of MIMIR and the machine it runs on.

Read-only by construction: the database is opened ``mode=ro``, the model
runtime is polled with ``/api/ps`` and ``/api/tags`` rather than a generate
call, and nothing here writes to the store it observes. A monitor that can
perturb a benchmark is not a monitor.
"""

from mimir.monitor.dashboard import run

__all__ = ["run"]
