"""labeler — combined, extensible annotation tool (FastAPI backend).

The package is served with::

    pixi run python -m tools.labeler

from the repository root. The frontend lives in ``tools/labeler/frontend``
(Vite + Svelte + TypeScript) and is built into ``frontend/dist``, which the
backend serves at ``/`` together with the JSON API under ``/api``.
"""

__version__ = "0.1.0"
