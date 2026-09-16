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

**Goal:** move the settings that differ per machine out of the code and into
environment variables, and make the app honest about whether it can actually
serve.

**Done:**

- Paths and tunables in `src/config.py` now come from environment variables via
  `pydantic-settings`, each defaulting to the value the code used before.
- `OPENAI_API_KEY` is required with no default, so the process stops at startup
  instead of failing on the first question.
- `CMD` and `HEALTHCHECK` read `${PORT:-8000}` so the host can pick the port.
- Built `/ready`: 503 when the vector store is empty or unreachable, 200 with a
  document count when the app can actually answer.

**Verified:** the container with no key exits with
`Configuration error: required environment variable(s) not set: OPENAI_API_KEY`.
With the key it starts and serves. `-e PORT=9000` moves the listener.

Note the distinction: the **image built fine** in both cases. It was the
**container** that refused to start — the image is the sealed box, the container
is that box running with an environment attached.

**What broke:** `index_size()` populates the module-level `_vectorstore`
singleton so `/ready` can count documents cheaply. But `_ensure_loaded()` — which
builds the BM25 index — decided it had nothing left to do by checking that same
variable. So a `/ready` probe arriving before the first `/ask` left `_bm25`
unbuilt and sparse retrieval dead.

**How I diagnosed it:** order-dependent, which is what made it dangerous. Calling
`/ask` first works, and that's what I do by hand. A container calls `/ready`
first, every 30 seconds, from startup. It would never have failed locally and
would have surfaced as "retrieval is broken in production" after deploying.

**Fix:** guard on what the function actually builds, not on a variable another
function now writes — `if _bm25 is not None: return`. Also: `/ready` answers 503
on every failure path rather than raising a 500, and the Docker `HEALTHCHECK` was
still probing `/health`, so Docker would have called an empty-index container
healthy.

**Regression test for Day 7:** probe `/ready` before the first `/ask`, assert the
answer still has citations.

**Time:** TODO
