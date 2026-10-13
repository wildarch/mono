"""Task modules for the provisioning tool.

Importing this package imports every task module, which registers all tasks as
a side effect.  To add new tasks, create a new module in this package and add
it to the imports below.
"""

from . import repos  # noqa: F401
from . import packages  # noqa: F401
from . import vscode  # noqa: F401
from . import gnome  # noqa: F401
from . import system  # noqa: F401
from . import speaker  # noqa: F401
from . import oracle  # noqa: F401
