"""
Tool-to-keyword mapping for intelligent tool pre-filtering.
"""
from __future__ import annotations


_TOOL_KEYWORD_MAP: dict[str, list[str]] = {


    #  -  -  Web & Research  -  - 


    "web_search": ["search", "find", "look up", "latest", "current", "news", "google", "bing", "duckduckgo", "what is", "who is", "where is", "how to", "define"],


    "fetch_url": ["read", "scrape", "fetch", "url", "website", "page", "article", "http", "extract", "parse html", "get content"],


    "http_request": ["http", "request", "api", "rest", "endpoint", "post", "get", "put", "patch", "delete", "curl", "fetch json"],


    "deep_research": ["research", "deep dive", "comprehensive", "thorough", "investigate", "analyze in depth", "detailed analysis", "full report", "study"],


    #  -  -  Browser  -  - 


    "browser_open": ["browse", "navigate", "open site", "visit", "screenshot", "webpage", "load page", "go to", "show website", "browser", "test browser", "your browser", "open browser", "try browser", "check browser", "view website", "see website"],


    "browser_screenshot": ["screenshot", "capture", "snap", "visual", "screenshot of", "capture screen", "take photo of page", "youtube", "video site"],


    "browser_click": ["click", "button", "link", "interact", "press", "select", "element"],


    "browser_fill_form": ["fill", "form", "input", "type into", "submit", "login", "sign in", "register", "type", "填写"],


    #  -  -  Code & Shell  -  - 


    "shell_run": ["run", "execute", "command", "terminal", "shell", "install", "build", "npm", "pip", "cargo", "go", "run command", "bash", "cmd",


                              "nextjs", "next.js", "react", "vue", "angular", "svelte", "nuxt", "gatsby",


                              "node", "express", "fastify", "koa", "hono",


                              "prisma", "drizzle", "mongoose", "sequelize",


                              "tailwind", "bootstrap", "chakra", "mui", "material ui",


                              "mongodb", "postgres", "mysql", "sqlite", "redis",


                              "graphql", "rest api", "api route", "trpc",


                              "component", "page", "layout", "middleware", "serverless",


                              "vercel", "netlify", "docker", "compose",


                              "webpack", "vite", "esbuild", "rollup",


                              "typescript", "tsx", "jsx", "css module",


                              "fullstack", "full-stack", "frontend", "backend",


                              "npm run", "yarn", "pnpm", "npx",


                              "deploy", "build app", "start app", "dev server"],


    "shell_sessions": ["sessions", "list sessions", "active session", "running commands"],


    "shell_reset": ["reset", "clear session", "kill session", "restart shell", "new session"],


    "python_exec": ["python", "code", "calculate", "compute", "script", "program", "simulate", "run python", "eval", "execute code", "计 -"],


    "code_explain": ["explain", "understand", "what does", "explain code", "analyze code", "what is this"],


    "code_write": ["write code", "create function", "implement", "build function", "coding"],


    #  -  -  Files  -  - 


    "file_read": ["read", "open", "show", "view", "file", "content", "display", "cat", "type", "get content", "查看"],


    "file_write": ["write", "save", "create", "generate", "export", "file", "make file", "new file", "写 - ", "创建"],


    "file_list": ["list", "files", "directory", "folder", "ls", "tree", "show files", "dir", "find file", "all files"],


    "file_delete": ["delete", "remove", "file", "clean", "unlink", "rm", "erase"],


    "file_search": ["search in files", "grep", "find in files", "find text", "search text", "find content"],


    #  -  -  Git  -  - 


    "git_init": ["git init", "new repository", "repo", "initialize", "start git", "init repo"],


    "git_clone": ["git clone", "clone", "download repo", "download repository", "get repo"],


    "git_status": ["git status", "changes", "modified", "unstaged", "uncommitted", "check status", "what changed"],


    "git_diff": ["git diff", "difference", "compare", "changes", "see changes", "what is different"],


    "git_add": ["git add", "stage", "track", "git add .", "stage changes"],


    "git_commit": ["git commit", "save", "version", "snapshot", "commit changes", "save version"],


    "git_push": ["git push", "deploy", "upload", "remote", "send to github", "upload changes"],


    "git_pull": ["git pull", "update", "sync", "fetch remote", "download changes", "get latest"],


    "git_log": ["git log", "history", "past commits", "commit history", "past versions"],


    "git_branch": ["branch", "create branch", "switch branch", "checkout", "new branch", "checkout -b"],


    "git_merge": ["merge", "combine branches", "join branches", "merge branch"],


    #  -  -  GitHub  -  - 


    "github_list_repos": ["github", "repos", "repositories", "my repos", "my repositories", "list repos"],


    "github_get_repo": ["github repo", "repo info", "repo details", "repository info", "get repo"],


    "github_list_issues": ["github issues", "bug", "issue", "github bugs", "open issues", "list issues"],


    "github_create_issue": ["create issue", "report bug", "file issue", "raise issue", "new issue", "bug report"],


    "github_list_prs": ["pull request", "pr", "github pr", "merge request", "open pr", "list pr"],


    "github_create_pr": ["create pr", "create pull request", "open pr", "new pr", "submit pr"],


    "github_read_file": ["github file", "read file from github", "source code", "github raw"],


    "github_commits": ["github commits", "commit history", "git log", "recent commits"],


    #  -  -  Email  -  - 


    "email_send": ["email", "mail", "send", "message", "smtp", "send email", "compose"],


    "email_inbox": ["inbox", "read email", "check email", "mail", "imap", "unread email", "received"],


    "email_search": ["search email", "find email", "email query", "filter email"],


    "email_reply": ["reply", "respond", "answer email", "respond to email"],


    #  -  -  Memory & Journal  -  - 


    "remember": ["remember", "memorize", "save fact", "note", "keep in mind", "store", "memory", "remember that"],


    "recall": ["recall", "remind", "what do you remember", "my info", "user facts", "remember me", "what do you know about me"],


    "journal_add": ["journal", "diary", "log", "entry", "daily log", "write journal", "note down"],


    "journal_search": ["search journal", "find entry", "past logs", "journal search"],


    #  -  -  Goals & Tasks  -  - 


    "goal_create": ["goal", "target", "milestone", "objective", "new goal", "set goal"],


    "goal_list": ["goals", "list goals", "my goals", "targets", "objectives"],


    "goal_update": ["update goal", "complete goal", "finish goal", "mark done", "goal status"],


    #  -  -  Web Monitor  -  - 


    "watch_url": ["watch url", "monitor url", "check regularly", "track url", "periodic check", "watch website", "monitor website", "web change"],


    "watch_path": ["watch file", "monitor file", "track file changes", "file watcher", "detect file change", "watch directory"],


    #  -  -  System  -  - 


    "system_info": ["system", "info", "about", "version", "status", "system info", "diagnostics"],


    "settings_get": ["settings", "config", "preferences", "configuration", "get settings"],


    "settings_update": ["update settings", "change settings", "configure", "set config"],


    #  -  -  Memory & Journal (deduplicated  - "recall" handles general recall, "recall_memories" for semantic search)  -  - 


    "recall_memories": ["recall memories", "search memories", "semantic search", "memory search", "find memories", "what memories"],


    "journal_write": ["journal", "diary", "log entry", "write journal"],


    "journal_read": ["read journal", "past entries", "journal history"],


    "journal_mood_summary": ["mood", "emotions", "feelings", "mood trends"],


    "memory_set_permanent": ["permanent memory", "protect memory", "never forget"],


    #  -  -  Goals  -  - 


    "add_goal": ["goal", "objective", "target", "plan", "todo"],


    "list_goals": ["goals", "objectives", "targets", "my goals", "list goals"],


    "update_goal_progress": ["progress", "update goal", "complete goal", "goal status"],


    #  -  -  System & Monitoring  -  - 


    "system_status": ["status", "health", "system info", "uptime"],


    "recent_events": ["events", "activity", "recent", "log", "history"],


    "schedule_task": ["schedule", "remind", "cron", "recurring", "every", "timer"],


    "list_watches": ["watches", "file monitors", "active watches"],


    "remove_watch": ["remove watch", "stop watching"],


    "monitor_url": ["monitor url", "track website", "web change", "url change"],


    "schedule": ["schedule", "cron", "recurring task", "automate"],


    "list_schedules": ["schedules", "cron jobs", "recurring tasks"],

    "list_nl_schedules": ["nl schedules", "natural language schedule"],

    "remove_schedule": ["remove schedule", "cancel cron", "delete recurring"],

    "remove_nl_schedule": ["remove nl schedule", "delete natural language schedule"],


    "improvement_log": ["improvement", "learn", "patterns", "failures"],


    #  -  -  Utility  -  - 


    "current_time": ["time", "date", "now", "clock"],


    "calculator": ["calculate", "math", "compute", "arithmetic"],


    "random_choice": ["random", "pick", "choose", "decide", "coin flip"],


    "generate_uuid": ["uuid", "unique id", "identifier", "random id"],


    #  -  -  3D Engine (Nexus3D)  -  - 


    "nexus3d_create_mesh": ["3d", "mesh", "cube", "sphere", "cylinder", "cone", "torus", "primitive", "3d model", "geometry", "vertex", "face", "obj"],


    "nexus3d_transform_mesh": ["transform", "translate", "rotate", "scale", "matrix", "4x4", "mesh transform"],


    "nexus3d_csg_boolean": ["csg", "boolean", "union", "subtract", "intersect", "carve", " hollow", "combine meshes"],


    "nexus3d_create_armature": ["armature", "skeleton", "rig", "bone", "joint", "rigging", "character"],


    "nexus3d_solve_ik": ["ik", "inverse kinematics", "solve ik", "armature"],


    "nexus3d_animate_procedural": ["animate", "animation", "walk", "run", "idle", "breathe", "procedural animation", "keyframe"],


    "nexus3d_render_scene": ["render", "rendering", "render scene", "png", "output image"],


    "nexus3d_path_trace": ["path trace", "path tracing", "ray trace", "production render"],


    "nexus3d_material_library": ["material", "pbr", "texture", "metal", "roughness", "material library"],


    "nexus3d_cinematic_dof": ["dof", "depth of field", "focus", "blur", "cinematic", "camera"],


    "nexus3d_physics_simulate": ["physics", "simulate", "collision", "gravity", "rigid body", "force"],


    "nexus3d_raycast": ["raycast", "ray", "intersect", "hit test"],


    #  -  -  Virtual Computer  -  - 


    "vm_start": ["start vm", "start virtual", "launch vm", "boot vm", "start workspace", "start desktop", "start computer", "start docker", "turn on vm"],


    "vm_stop": ["stop vm", "stop virtual", "shutdown vm", "halt vm", "stop workspace", "shutdown desktop", "stop computer", "turn off vm"],


    "vm_restart": ["restart vm", "reboot vm", "restart virtual", "reboot desktop"],


    "vm_execute": ["execute", "run command", "terminal", "shell", "command", "install", "run script", "bash", "cmd", "exec", "virsh", "docker", "systemctl", "apt", "yum", "pip", "npm", "python", "node", "sudo"],


    "vm_screenshot": ["screenshot", "capture", "screen", "vm screenshot", "virtual screen"],


    "vm_mouse": ["mouse", "click", "move mouse", "cursor"],


    "vm_keyboard": ["keyboard", "type", "key press", "type text"],


    "vm_vision_loop": ["vision loop", "ai vision", "visual automation", "gui automation", "computer use", "use computer", "visual agent", "autonomous control"],


    "vm_status": ["vm status", "virtual status", "check vm", "docker status"],


    #  -  -  PyTorch / ML  -  - 


    "pt_model_create":       ["create model", "neural network", "build model", "architecture", "layers", "pytorch", "define model", "nn module"],


    "pt_model_train":        ["train", "training", "fit model", "epochs", "batch", "loss", "optimizer", "train model", "ml training"],


    "pt_model_predict":      ["predict", "inference", "classify", "forward pass", "run model", "model prediction"],


    "pt_model_evaluate":     ["evaluate", "accuracy", "metrics", "test model", "benchmark", "model evaluation"],


    "pt_model_save":         ["save model", "checkpoint", "export model", "serialize", "persist model"],


    "pt_model_load":         ["load model", "restore", "import model", "deserialize", "resume training"],


    "pt_hyperparameter_tune":["hyperparameter", "tuning", "grid search", "optimize params", "hp tune"],


    "pt_transfer_learning":  ["transfer learning", "pretrained", "fine-tune", "resnet", "vgg", "efficientnet", "finetune"],


    "pt_text_classification":["text classification", "sentiment", "nlp", "classify text", "text classifier"],


    "pt_image_classification":["image classification", "cv", "computer vision", "classify image", "image classifier"],


    "pt_model_convert":      ["onnx", "torchscript", "convert model", "deploy model", "export"],


    #  -  -  MCP  -  - 


    "mcp_connect_server":    ["mcp", "connect server", "external tool", "plugin", "mcp connect"],


    "mcp_call_tool":         ["mcp tool", "call tool", "external service", "mcp call", "mcp invoke"],


    "mcp_auto_connect":      ["auto connect", "mcp auto", "discover tools", "auto discover"],


    #  -  -  Integration Tools  -  - 


    "stripe_create_payment": ["stripe", "payment", "charge", "billing", "invoice", "pay"],


    "github_create_issue":   ["github issue", "bug report", "create issue", "file issue"],


    "slack_send_message":    ["slack", "send message", "notify", "channel", "slack message"],


    "notion_create_page":    ["notion", "page", "wiki", "database", "notion page"],


    "supabase_query":        ["supabase", "database", "postgres", "query", "sql query"],


    #  -  -  Presentation Tools  -  - 


    "create_ppt":            ["ppt", "powerpoint", "presentation", "slide", "slides", "deck", "present", "make presentation", "create slides", "generate presentation", "presentation file", "pptx", "power point"],


    "ppt_add_slide":         ["add slide", "new slide", "append slide", "extra slide", "additional slide", "add content", "add section", "more slides", "another slide"],


    "list_ppt_templates":    ["template", "theme", "style", "presentation theme", "slide theme"],


    #  -  -  Document Tools (v44)  -  - 


    "docx":                  ["docx", "word", "word document", "word doc", "microsoft word", "create document", "create doc", "make document", "write document", "generate document", "report", "formal document", "doc file", ".docx", "professional document", "document with sections", "word report", "word file"],


    "list_docx_templates":   ["docx template", "document theme", "word template", "document template", "docx theme"],


    #  -  -  Full-Stack Development (v26)  -  - 


    # (keywords merged into shell_run above to avoid dict key overwrite)


    #  -  -  Project Context (v39)  -  - 


    "project_read_context":  ["project context", "read project", "project.md", "load project", "project notes", "project goals"],


    "project_write_context": ["save project", "write project", "update project", "project.md", "persist project", "project state"],


    #  -  -  Delegate / Sub-agents (v40)  -  - 


    "delegate_task": ["delegate", "spawn sub-agent", "use sub-agent", "delegate task", "parallel task", "split work", "do in background", "parallel", "concurrent"],


    "delegate_batch": ["delegate multiple", "spawn sub-agents", "parallel work", "batch delegate", "multiple tasks", "create multiple files", "create several files", "build files in parallel", "run in parallel", "parallel execution", "multiple files"],

    #  -  -  Vision / Image Analysis  -  - 

    "image_understand": ["image", "photo", "picture", "screenshot", "analyze image", "what is in this image", "describe image", "vision", "image content", "read image", "image analysis", "image recognition", "visual", "see what", "look at this", "view image", "image file", "understand image", "visual content", "recognize"],


}


# Skills that the agent can automatically use when needed


# These are distinct from tools  - skills are high-level capabilities


# that may combine multiple tools or require specialized knowledge.


