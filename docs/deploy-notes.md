# Deploy notes — Layer 8

One entry per day. The point of this file is the "how I diagnosed it" line —
what the logs said and how I narrowed it down, not just the fix.

---

## Day 1 — 2026-09-11

**Goal:** check that the project's docs match the code, pin the dependencies,
and record a baseline image size to measure later work against.

**What broke:** three failed Docker builds before the first green one. Also
found four claims in `CLAUDE.md` the code doesn't support (a reranker, per-
citation verification, LangSmith tracing, a status log stuck at v0.3), and a CV
listing FAISS and PostgreSQL when the project uses Chroma and SQLite.

**How I diagnosed it:**

The real failure was the second build. `requirements.txt` listed minimum
versions — "langchain 0.3 or newer", "langgraph 0.2 or newer" — with no upper
limit. My laptop had a set that works (langgraph 1.2.7, langchain 1.3.11),
but I'd installed those one at a time over months. A fresh container has to pick
every version at once from the list, and no combination fit: every recent
langchain requires langgraph below 1.3.0, while an open-ended "0.2 or newer"
kept selecting something higher. pip spent seven minutes trying ~40
combinations and gave up with `ResolutionImpossible`.

The other two failures weren't the project. Build 1 was one package's download
timing out (langgraph had downloaded fine seconds earlier, so it wasn't the
network as a whole). Build 3 was a download dying halfway — the error came from
pip's download step, after version-picking had already succeeded.

**Fix:** replaced every minimum in `requirements.txt` with the exact version my
venv runs, so there's nothing left to figure out. Moved the eval-only packages
(ragas, datasets) into `requirements-dev.txt` so they never enter the runtime
image. Switched the base image to `python:3.12-slim` to match my venv. Gave pip
a longer timeout and a build cache, so a failed download is retried from where
it stopped instead of from zero — that alone took the next build from 24
minutes to under 4.

**Baseline:** `rag-app:before` — 960MB on disk, 199MB compressed. 593MB of that
is the single dependency-install layer.

**Time:** ~3h, most of it waiting on failed builds.

---

## Day 2 — 2026-09-15

**Goal:** lock the dependencies properly, and find out whether a two-stage build
and removing an unused package actually shrink the image.

**What broke:** a `docker run` command failed with
`exec: "#": executable file not found`. And the new image came out the same size
as the old one, which shouldn't happen if the build had really changed.

**How I diagnosed it:**

The `#` error meant the container tried to run a program called `#`. I'd pasted
a command with a trailing comment, and zsh doesn't strip `#` comments when
you're typing interactively — so the comment became arguments.

The identical sizes: I read the Dockerfile back and it was still the old
single-stage one. I'd generated the lockfile but never pointed the build at it,
so the "new" image was just the old image minus one package. Check the input
changed before drawing conclusions from the output.

**Fix:** dropped `langchain` — nothing imports it (the code uses
`langchain-core`, `-community`, `-openai`, `-chroma`, `-text-splitters`
directly) and `pip show langchain` confirmed nothing depends on it. Generated
`requirements.lock.txt` by running `pip freeze` *inside* the built image, so it
records what actually got installed rather than what my Mac happens to have.
Rewrote the Dockerfile in two stages: one installs the packages, the other
copies only the finished result.

**Result:**

| image | disk | compressed |
|---|---|---|
| `rag-app:before` | 960MB | 199MB |
| `rag-app:trial` (no `langchain`) | 958MB | 199MB |
| `rag-app:after` (two-stage + lockfile) | 963MB | 199MB |

Honest read: **the two-stage build made the image 3MB bigger, not smaller.**
Dropping `langchain` saved 2MB; multi-stage then added 5MB back. The venv at
`/opt/venv` carries its own copy of pip and setuptools, while the slim base image
already has a system copy — so the runtime image now ships both. Compressed size
never moved at all: 199MB before, during and after, which is the number that
matters for pushing to ECR and pulling onto EC2.

The only real reduction in this whole exercise happened on Day 1, by splitting the
eval packages out of `requirements.txt` — and that was done to fix a build failure,
not as an optimisation. The remaining 593MB is chromadb's ONNX and gRPC
dependencies, which cannot go while chromadb stays.

I kept the two-stage build anyway, for the lockfile install and as a place to put a
compiler if a package ever needs to build from source. Not for size. Claiming
"reduced image size with a multi-stage build" here would be false.

**Time:** TODO

---

## Day 3 — 2026-09-15

**Goal:** move every setting out of the code and into environment variables, make
missing configuration stop the app at startup instead of at request time, and let
the host decide which port to listen on.

**What broke:**

`curl localhost:9000/health` failed with `Failed to connect ... Couldn't connect
to server`, right after a container that had started perfectly well.

**How I diagnosed it:**

The failure looked like a port-mapping mistake, which is what I went looking for
first. It wasn't. `docker run` without `-d` runs in the foreground and holds the
terminal, so I had pressed Ctrl+C to get my prompt back — which stopped the
container. By the time curl ran there was genuinely nothing listening. The fix is
`-d` plus `docker logs`, which is also how the app will run on a real host: no
server runs attached to a terminal.

Worth recording separately: **the fail-fast behaviour cannot be tested on my Mac.**
`load_dotenv()` reads `.env`, so unsetting `OPENAI_API_KEY` in the shell puts it
straight back. The only honest test is in the container, where `.dockerignore`
excludes `.env` and there is no key to find. Same shape as the Day 1 lesson: my
laptop carries state the container doesn't.

**Fix:**

- `src/config.py` rewritten on `pydantic-settings`. Every constant became a field
  whose default is the old hardcoded value, so behaviour is unchanged unless
  something is explicitly overridden. The module still exports the same
  upper-case names, so no other file changed.
- `openai_api_key: str` with no default is the whole fail-fast mechanism —
  `Settings()` raises at import, before uvicorn binds a port. A `try/except`
  turns pydantic's error into a plain message naming the missing variable.
- `TAVILY_API_KEY` stays optional: web search degrades, the app still serves.
- `load_dotenv()` kept deliberately. `langchain-openai` reads `OPENAI_API_KEY`
  from `os.environ` directly, not from the Settings object. Removing it would
  produce an app that starts cleanly, validates happily, and then fails on the
  first request with an auth error.
- `CMD` changed from JSON form to `sh -c "exec uvicorn ... --port ${PORT:-8000}"`.
  JSON form doesn't expand variables, so `$PORT` would have been passed as a
  literal string. `exec` makes uvicorn replace the shell and become PID 1, so it
  receives Docker's SIGTERM itself — without it the shell swallows the signal and
  the process is hard-killed ten seconds later, mid-request.
- `HEALTHCHECK` also read `${PORT:-8000}` instead of a second hardcoded 8000. The
  port is decided in one place and everything that mentions it reads from there.
- `.env.example` rewritten to list every variable with its default.

**Verified:**

| test | result |
|---|---|
| container with no key | exits with `Configuration error: required environment variable(s) not set: OPENAI_API_KEY` |
| container with key | starts, `/health` returns 200 |
| `CHROMA_DIR=/tmp/somewhere` locally | config reports the overridden path |
| `-e PORT=9000` | uvicorn logs `running on http://0.0.0.0:9000` |

**Still open:** `/ready` — the endpoint that reports whether the index is actually
loaded — is not built yet.

**Time:** TODO
