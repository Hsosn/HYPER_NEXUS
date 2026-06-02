# Contributing to Hyper Nexus

Thanks for your interest in making Hyper Nexus better. This document covers the
mechanics of contributing. The **.github/ISSUE_TEMPLATE** and
**.github/PULL_REQUEST_TEMPLATE** describe what we expect in a good report.

---

## Code of Conduct

Everyone who participates in this project is expected to follow the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Be kind, be patient,
assume good faith.

---

## What we are looking for

The fastest way to be useful:

| Area | What helps |
|---|---|
| **Bug reports** | A failing test, a stack trace, the exact commit hash, and the `.env` keys you set (never paste the values). |
| **Tool / integration implementations** | New entries in `nexus/integrations/*.py` with a real `service_id`, OAuth or API-key auth, and at least one end-to-end test. |
| **Reasoning modules** | Anything in `nexus/reasoning/` (ADHD, reflection, self-supervised) is welcomed — but **disable it by default** and gate it behind a config flag, otherwise it changes the agent's behaviour for everyone. |
| **Skills** | Drop a folder with a `SKILL.md` into `nexus3d_skills/` or `nexus_ml_skills/` following the format of the existing skills. |
| **Documentation** | Typos, broken examples, unclear wording in `README.md`, `docs/`, or docstrings. |
| **Tests** | The project currently has zero automated tests. Adding any is a massive contribution. |

We are **not** looking for:
* Style-only refactors with no behaviour change.
* Brand-new dependencies unless the benefit clearly outweighs the install cost.
* Features that require a paid service to be useful by default.

---

## Development setup

```bash
# 1. Clone
git clone https://github.com/<you>/hyper-nexus.git
cd hyper-nexus

# 2. Python 3.11+ (3.12 recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. Start Redis (Celery broker + event bus)
docker compose up -d redis

# 4. Copy and edit your env file
cp .env.example .env
# put at least one LLM key in .env

# 5. Run the server
uvicorn nexus.api.server:app --reload --port 8000

# 6. In a second terminal, run the Celery worker + beat
celery -A nexus.celery_app worker -l info
celery -A nexus.celery_app beat -l info
```

The WebUI is served at <http://localhost:8000/webui/>. The OpenAPI docs are at
<http://localhost:8000/docs>.

---

## Project layout

```
nexus/
├── api/              FastAPI routes, WebSocket endpoints
├── reasoning/        Engine, ADHD module, reflection, self-supervised learning
├── tools/builtin/    File, shell, web, code, browser tools
├── integrations/     100+ service adapters (one file per provider family)
├── tasks/            Celery tasks (heartbeat, scheduler, self-improvement)
├── memory/           SQLite-backed long-term memory
└── config.py         Pydantic settings, defaults, env-var bindings
webui/                Static SPA (vanilla JS, no build step)
nexus_ml_skills/      13 ML skill docs (PyTorch, JAX, diffusion, etc.)
nexus3d_skills/       12 3D skill docs (Blender, USD, Gaussian splatting, etc.)
```

---

## How to make a change

1. **Open an issue first** for non-trivial changes. A 2-sentence "I'm planning
   to do X because Y" message saves everyone time.
2. **Branch off `main`.** Use a descriptive name: `fix/sandbox-escape-on-windows`
   or `feat/add-notion-integration`.
3. **Keep PRs small and focused.** One change per PR. A 200-line PR is good; a
   2000-line PR is a code review nightmare.
4. **Match the existing style.** Run `ruff check nexus/` (we will add a
   pre-commit hook in the next release — until then, eyeball it).
5. **Add or update a test** if you fix a bug. The test should fail *without*
   your change and pass *with* it.
6. **Update the README** if your change is user-visible.
7. **Never commit secrets.** Use the encrypted-credentials API (`/api/credentials`)
   to store keys at rest.

---

## Commit messages

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(integrations): add Notion adapter
fix(sandbox): block writes outside data/workspace on Windows
docs: clarify OPENROUTER_API_KEY in README
refactor(reasoning): split ADHD module into domain registry + dispatcher
test(tools): cover file sandbox rejection paths
```

---

## Pull request checklist

* [ ] I have read the [Code of Conduct](CODE_OF_CONDUCT.md).
* [ ] I have opened (or commented on) an issue describing the change.
* [ ] My branch is up-to-date with `main`.
* [ ] `uvicorn nexus.api.server:app` boots without errors.
* [ ] I have not committed any secrets, `.env` files, or API keys.
* [ ] I have updated the README / docs if my change is user-visible.
* [ ] I have not added a new top-level dependency without justification.

---

## Reporting security issues

**Please do not open a public issue.** Email the maintainer directly (see
`MAINTAINERS.md` once it lands, or the GitHub profile of the most active
committer). We will respond within 72 hours and credit you in the fix.

---

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
