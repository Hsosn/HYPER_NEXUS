from . import config
from . import environment  # noqa: F401 — Environment Awareness module (always active by default)

# Core tool imports that should always be available.
# These trigger the @tool decorator side-effects which register each tool.
from .tools.builtin import (  # noqa: F401
    basic_tools, custom_loader,
    email_tools, file_tools, git_tools, github_tools,
    goal_tools, memory_tools, monitor_tools,
    research_tools, shell_session, system_tools, web_tools,
    journal_tools, integration_tools,
)

# Optional tool modules — may be skipped by builtin/__init__.py if their
# optional dependencies (e.g., playwright, torch, ffmpeg) are missing.
# Wrap each in try/except to prevent a missing optional dep from crashing
# the entire application on startup.
try:
    from .tools.builtin import code_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import browser_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import nexus3d_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import presentation_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import ffmpeg_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import pytorch_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import virtual_computer_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import image_gen_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import mcp_tools  # noqa: F401
except ImportError:
    pass
try:
    from .tools.builtin import vision_tools  # noqa: F401 — Vision / image analysis tools
except ImportError:
    pass
try:
    from .tools.builtin import ml_ai_skill_tools  # noqa: F401 — ML/AI skill bridge tools (13 skills)
except ImportError:
    pass
try:
    from .tools.builtin import fullstack_tools  # noqa: F401 — Full-stack dev scaffolding & templates
except ImportError:
    pass
try:
    from .tools.builtin import docx_tools  # noqa: F401 — Word document creation tools
except ImportError:
    pass
from .tools.builtin import delegate_tools  # noqa: F401 — Sub-agent delegation tools

from .tools.registry import REGISTRY
from .events import emit
