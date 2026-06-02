"""Register built-in tools."""
from . import basic_tools    # noqa: F401
from . import web_tools      # noqa: F401
from . import file_tools     # noqa: F401
try:
    from . import code_tools     # noqa: F401
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("code_tools not available: %s", e)
from . import memory_tools   # noqa: F401
from . import goal_tools     # noqa: F401
from . import system_tools   # noqa: F401
from . import monitor_tools  # noqa: F401
try:
    from . import browser_tools  # noqa: F401
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("browser_tools not available: %s", e)
from . import git_tools      # noqa: F401
from . import shell_session  # noqa: F401
from . import email_tools    # noqa: F401
from . import github_tools   # noqa: F401
from . import research_tools # noqa: F401
from . import custom_loader  # noqa: F401
try:
    from . import ffmpeg_tools  # noqa: F401
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("ffmpeg_tools not available: %s", e)
try:
    from . import nexus3d_tools  # noqa: F401
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("nexus3d_tools not available: %s", e)
from . import journal_tools   # noqa: F401
from . import integration_tools  # noqa: F401
try:
    from . import pytorch_tools  # noqa: F401 — ML/AI deep learning tools (PyTorch)
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("pytorch_tools not available: %s", e)
try:
    from . import presentation_tools  # noqa: F401 — PowerPoint tools
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("presentation_tools not available: %s", e)
try:
    from . import virtual_computer_tools  # noqa: F401
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("virtual_computer_tools not available: %s", e)
try:
    from . import image_gen_tools  # noqa: F401
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("image_gen_tools not available: %s", e)
try:
    from . import mcp_tools       # noqa: F401 — MCP server management tools
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("mcp_tools not available: %s", e)
try:
    from . import ml_ai_skill_tools  # noqa: F401 — ML/AI skill bridge tools (13 skills)
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("ml_ai_skill_tools not available: %s", e)
from . import delegate_tools  # noqa: F401 — Sub-agent delegation tools
try:
    from . import fullstack_tools  # noqa: F401 — Full-stack dev scaffolding & templates
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("fullstack_tools not available: %s", e)
try:
    from . import docx_tools  # noqa: F401 — Word document creation tools (python-docx)
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("docx_tools not available: %s", e)
from . import vision_tools  # noqa: F401 — Vision / image analysis tools
try:
    from . import vision_loop  # noqa: F401 — Autonomous GUI vision loop (requires VM)
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("vision_loop not available: %s", e)
