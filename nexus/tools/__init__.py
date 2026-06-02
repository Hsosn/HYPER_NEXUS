from .builtin import (  # noqa: F401 — triggers tool registration side-effects
    basic_tools, custom_loader,
    email_tools, file_tools, git_tools, github_tools,
    goal_tools, memory_tools, monitor_tools,
    research_tools, shell_session, system_tools, web_tools,
    journal_tools, integration_tools,
)
# Optional modules — wrapped in try/except because builtin/__init__.py
# may skip them if their dependencies are missing.
try:
    from .builtin import code_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import browser_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import nexus3d_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import presentation_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import ffmpeg_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import pytorch_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import virtual_computer_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import image_gen_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import mcp_tools  # noqa: F401
except ImportError:
    pass
try:
    from .builtin import vision_tools  # noqa: F401 — Vision / image analysis tools
except ImportError:
    pass
try:
    from .builtin import ml_ai_skill_tools  # noqa: F401 — ML/AI skill bridge tools (13 skills)
except ImportError:
    pass
try:
    from .builtin import fullstack_tools  # noqa: F401 — Full-stack dev scaffolding & templates
except ImportError:
    pass
try:
    from .builtin import docx_tools  # noqa: F401 — Word document creation tools
except ImportError:
    pass
from .builtin import delegate_tools  # noqa: F401 — Sub-agent delegation tools
from .registry import REGISTRY
