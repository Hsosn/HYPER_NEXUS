"""
Available skills catalog for the Nexus reasoning engine.
"""
from __future__ import annotations


SKILLS = [


    #  -  -  AI & Media  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "generate_image", "desc": "Generate AI images from text prompts", "cat": "AI & Media"},


    {"name": "video_generation", "desc": "Generate AI videos from text prompts", "cat": "AI & Media"},


    {"name": "video_understand", "desc": "Analyze and understand video content frame-by-frame", "cat": "AI & Media"},


    #  -  -  Research  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "web_search", "desc": "Search the web for real-time information", "cat": "Research"},


    {"name": "fetch_url", "desc": "Extract and read content from any web page", "cat": "Research"},


    {"name": "deep_research", "desc": "Multi-query parallel research with source synthesis", "cat": "Research"},


    #  -  -  Document Creation  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "pdf", "desc": "Create professional PDF documents", "cat": "Documents"},


    {"name": "docx", "desc": "Create Word documents", "cat": "Documents"},


    {"name": "xlsx", "desc": "Create Excel spreadsheets with formulas and charts", "cat": "Documents"},


    {"name": "ppt", "desc": "Create PowerPoint presentations with multiple slide types (title, content, two-column, image, blank)", "cat": "Documents"},


    {"name": "charts", "desc": "Generate charts, diagrams, and data visualizations", "cat": "Documents"},


    #  -  -  Development  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "fullstack_dev", "desc": "Full-stack web development (Next.js, React, API, DB)", "cat": "Development"},


    {"name": "agent_browser", "desc": "Browser automation  - navigate, click, type, screenshot", "cat": "Development"},


    {"name": "code_execution", "desc": "Run Python code and shell commands", "cat": "Development"},


    {"name": "git_workflow", "desc": "Git version control  - clone, commit, push, PR", "cat": "Development"},


    #  -  -  Finance  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "finance", "desc": "Financial data analysis  - stocks, markets, portfolios", "cat": "Finance"},


    #  -  -  3D Engine  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "nexus3d_create_mesh", "desc": "Create 3D meshes (sphere, cube, cylinder, etc.)", "cat": "3D Engine"},


    {"name": "nexus3d_csg_boolean", "desc": "Constructive solid geometry - boolean operations on meshes", "cat": "3D Engine"},


    {"name": "nexus3d_animate_procedural", "desc": "Generate procedural animations - walk, idle, breathe, sine wave", "cat": "3D Engine"},


    #  -  -  ML / AI Engineering  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "transformer_architect", "desc": "Design and configure transformer model architectures", "cat": "ML Engineering"},


    {"name": "llm_trainer", "desc": "Fine-tune and train LLMs with custom datasets", "cat": "ML Engineering"},


    {"name": "rlhf_lab", "desc": "Reinforcement Learning from Human Feedback workflows", "cat": "ML Engineering"},


    {"name": "peft_finetuning", "desc": "Parameter-efficient fine-tuning (LoRA, QLoRA, Adapters)", "cat": "ML Engineering"},


    {"name": "deep_learning_trainer", "desc": "Train deep learning models with mixed precision", "cat": "ML Engineering"},


    {"name": "nlp_workbench", "desc": "NLP tasks  - tokenization, NER, sentiment, summarization", "cat": "ML Engineering"},


    {"name": "cv_workbench", "desc": "Computer vision  - classification, detection, segmentation", "cat": "ML Engineering"},


    {"name": "data_pipeline", "desc": "Build data processing and ETL pipelines", "cat": "ML Engineering"},


    {"name": "model_optimizer", "desc": "Optimize models  - quantization, pruning, distillation", "cat": "ML Engineering"},


    {"name": "neural_architect", "desc": "Neural architecture search and design", "cat": "ML Engineering"},


    {"name": "gan_studio", "desc": "Generative adversarial networks  - image synthesis", "cat": "ML Engineering"},


    {"name": "distributed_training", "desc": "Distributed multi-GPU training strategies", "cat": "ML Engineering"},


    {"name": "rl_lab", "desc": "Reinforcement learning environments and agents", "cat": "ML Engineering"},


    #  -  -  Media Processing  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 


    {"name": "ffmpeg_convert", "desc": "Convert media between formats (video, audio)", "cat": "Media"},


    {"name": "ffmpeg_extract_audio", "desc": "Extract audio track from video file", "cat": "Media"},


    {"name": "ffmpeg_create_gif", "desc": "Create animated GIF from video clip", "cat": "Media"},


    {"name": "ffmpeg_trim", "desc": "Trim video to specific time range", "cat": "Media"},


    {"name": "ffmpeg_merge_video", "desc": "Merge/concatenate multiple video clips", "cat": "Media"},


    {"name": "ffmpeg_add_subtitles", "desc": "Add subtitles to video", "cat": "Media"},


]


