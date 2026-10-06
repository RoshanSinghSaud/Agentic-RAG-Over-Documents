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
`requirements.lock.txt` by running `pip freeze` _inside_ the built image, so it
records what actually got installed rather than what my Mac happens to have.
Rewrote the Dockerfile in two stages: one installs the packages, the other
copies only the finished result.

**Result:**

| image                                  | disk  | compressed |
| -------------------------------------- | ----- | ---------- |
| `rag-app:before`                       | 960MB | 199MB      |
| `rag-app:trial` (no `langchain`)       | 958MB | 199MB      |
| `rag-app:after` (two-stage + lockfile) | 963MB | 199MB      |

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

---

## Day 4 — 2026-09-16

**Goal:** a brand-new container, with no saved search index, should build the
index by itself at startup and then answer questions.

**Done:**

- The papers now ship inside the image instead of being downloaded at startup.
- On startup the app checks whether the index is empty. If it is, it builds it
  before accepting any request. If not, it skips straight to serving.
- Added `scripts/check_day4.sh`, which tests this from scratch in one command.

**What broke (caught in review, before it ran):** the startup check opened the
index, and the build step then deleted the index folder while it was still
open. Chroma can't write to a folder deleted underneath it, so the first boot
on a fresh host would have crashed with a "readonly database" error. It never
showed up locally, because my laptop already has an index and the build step
never runs there.

**How I diagnosed it:** reproduced the same open → delete → rebuild order in a
small script against the exact Chroma version in the lockfile. Same error.

**Fix:** the startup build no longer deletes the folder — it's already empty,
which is why it's building. The manual `python main.py ingest` still wipes and
rebuilds as before. Also changed the "index is empty" error so it fails one
request instead of possibly stopping the whole app.

**Result:**

| check                     | result                                              |
| ------------------------- | --------------------------------------------------- |
| first start, empty index  | ready in 29s (718 chunks from 156 pages)            |
| test question             | correct answer, 6 citations from the Self-RAG paper |
| memory after one question | 294MB (Render free tier allows 512MB)               |
| restart                   | ready in 2s, index reused, no rebuild               |

**Known limitation — conversation memory:** chat history is saved in a small
file (`checkpoint.db`) inside the container. On a free host that sleeps when
idle or redeploys, that file is wiped, so conversations and any pending
"approve web search?" questions are lost. On Render this happens after 15
minutes without visitors, not only on redeploy. On EC2 (Day 16) the file will
live on the server's own disk and survive restarts.

**Still open:** log how long the build itself takes; handle a build that gets
cut off halfway (right now it would be treated as finished).

**Time:** TODO

---

## Day 5 — 2026-09-17

**Goal:** put the app on the internet, but only for people I give a password to.

**Live link:** https://agentic-rag-over-documents.onrender.com/docs
(free hosting — if nobody has used it for 15 minutes, the first request can take
about a minute while it wakes up)

**What I added:**

- **A password (API key).** Asking a question now needs a secret key. Without
  it, the app says "not authenticated". The status pages stay open so the
  hosting service can check the app is alive.
- **A speed limit.** Each visitor can send at most 10 requests a minute. This
  stops someone from running up my OpenAI bill.
- **One copy of the app at a time.** The chat memory and the speed limit both
  live inside the running app, so two copies would each keep their own and
  neither would work properly.
- **A test script** (`scripts/check_day5.sh`) that checks all of the above in
  one go, on my laptop or on the live link. 13/13 checks pass on both.

**What broke, and how I found it:**

1. **The live app would have had no papers to read.** The papers aren't stored
   on GitHub, and the hosting service builds from GitHub. My laptop tests all
   passed because my laptop has the papers. Testing a fresh copy of the code
   from GitHub showed the app had read only its own README — 2 text pieces
   instead of about 700 — and couldn't answer anything.
   _Fix:_ the papers are now downloaded while the app is being built. The test
   script now also fails if the app has too few text pieces.
2. **First deploy refused to start.** The log said the `API_KEY` setting was
   missing — I hadn't entered it on the hosting site. The app stopping with a
   clear message (instead of running unprotected) is what it was designed to do.
   _Fix:_ added the setting and redeployed.
3. **All visitors could have shared one speed limit.** Behind the hosting
   service, every visitor looks like they come from the same address, so one
   person could have locked everyone out.
   _Fix:_ told the app to read the visitor's real address. Checked from a
   different network: my limit and theirs are separate.
4. **I locked myself out.** Right after testing, my own questions got "rate
   limit exceeded". Not a bug — every request counts toward the 10, even
   rejected ones, and my test script had used them up. Waiting a minute fixed
   it.

**Results:**

| check                             | result      |
| --------------------------------- | ----------- |
| time to answer a question (live)  | 7.8 seconds |
| text pieces in the live app       | 692         |
| wake-up time after being idle     | TODO        |
| build time on the hosting service | TODO        |

**Still open:**

- The live app has 692 text pieces, my laptop 716 — the downloaded papers are
  slightly different versions from my local copies. Need to fix the exact
  versions so both always match.
- 10 requests a minute is tight for someone trying the demo; may raise it.

**Time:** 4 hrs

---

## Day 7 — 2026-09-18

**Goal:** start an actual `pytest` suite instead of only manual runs and the
`scripts/check_day*.sh` scripts — beginning with the two pieces that matter
most and are least exercised by hand: the startup guard Render's cold boot
depends on (`ensure_index`), and the step that merges the two retrievers'
results (nothing in the eval set isolates it from everything downstream).

**Done:**

- Test scaffolding: pinned `pytest`/`httpx`, a `pytest.ini`, and a
  `conftest.py` that fills in fake secrets before anything imports the app
  (`src/config.py` otherwise exits immediately with no `OPENAI_API_KEY`), plus
  two small sample documents in `tests/fixtures/` so tests never touch the real
  papers or spend real money.
- 11 tests across 5 files, covering every item on the roadmap's Day 7 list:
  chunker output shape, retriever caps at `TOP_K`, ingestion idempotency,
  `/ready` reporting the index, and a malformed request returning 4xx — plus
  the `ensure_index` startup guard as a sixth area not on the original list.

**What broke, and how I found it:**

1. **Building the index a second time in the same run crashed the app**,
   with the database reporting itself as read-only. This is the same family
   of bug as Day 4 — Chroma can't be written to while something has it
   deleted-and-reopened underneath it — but a different trigger: Day 4 was
   "delete the folder while it's still open," this was "the app remembers an
   old connection to a folder that no longer exists." Writing a test that
   simply builds the index twice in a row reproduced it immediately, for
   free, with two fake sample documents instead of the real 156-page corpus.
   _Fix:_ after the folder is deleted, also clear the app's memory of the old
   connection, so the next build opens a genuinely fresh one instead of
   reusing the stale one. Confirmed the test now goes from failing to passing
   with only that one line changed.

2. **My own test would have passed even if the merge step were broken.** To
   test "a passage found by both search methods shouldn't get listed twice,"
   I first wrote the test using one shared example object standing in for
   "the same passage" in both fake result lists. That's not how the real app
   works — the two search methods each independently produce their own
   separate description of the same passage, not one shared object. Because
   my test used a shortcut, it would have said "no duplicates" purely because
   it was literally the same object in memory, regardless of whether the
   actual matching logic worked at all. I caught this by deliberately
   breaking the matching logic and re-running the test — it still passed,
   which meant it wasn't testing anything real.
   _Fix:_ rebuilt the test with two separate objects holding matching
   content, the way the real search methods actually produce them. Broke the
   matching logic again to confirm the corrected test now fails the way it
   should, then restored the real code and confirmed the test suite passes
   against it, unmodified.

3. **The plan itself was already out of date.** The roadmap's checklist said
   "`/health` returns 200 and reports a non-empty index." But Day 3 had
   already split that in two: `/health` is unconditional liveness (always
   200, so the platform never kills a container that's merely still
   building), and `/ready` is the one that reports the document count and
   can fail. Writing the test against what the code actually does — not what
   the plan said it should do — surfaced the drift.
   _Fix:_ corrected the roadmap line instead of writing a test for a `/health`
   behavior that was never built.

4. **A naive "bad request" test would have proven the wrong thing.** `/ask`
   checks the API key and rate limit *before* it looks at the request body.
   Sending a bad request with no key at all fails on authentication first —
   the body is never even read, so the test would silently be checking
   `require_api_key`, not request validation. Sent a valid key with the
   `question` field missing instead, and asserted the specific code (422),
   not just "anything under 500" — the latter would pass just as happily if
   the key check were the thing returning 4xx.

**Result:** 11/11 tests passing (`ensure_index`: 2, ingest-twice regression:
1, retrieval merge + RRF: 2, chunker + loader: 2, app `/health` + `/ready` +
malformed request: 4).

**Interview talking points:**

- **The idempotency bug had a second layer the original fix never found.**
  The known story (README, v0.4) is "ran ingest four times, index quadrupled,
  fixed by wiping the folder first." What the regression test found is that
  the fix was incomplete: wiping the folder doesn't stop the database
  library from reusing an old, now-invalid connection to it if you rebuild
  *within the same running process* — that failed differently (a hard crash,
  not silent duplication) and would never have shown up in the original
  incident, which happened across separate command-line runs. Good answer to
  "tell me about a bug you fixed" that isn't just the bug — it's "I wrote the
  regression test for an old bug and it found a second, different bug hiding
  under the first one."
- **A test that passes for the wrong reason is worse than no test.** Caught
  my own dedup test silently checking Python object identity instead of the
  actual matching logic, because I'd built the fake data with a shortcut
  that doesn't reflect how the real code produces it. Confirmed by
  deliberately breaking the logic and watching the test wrongly stay green —
  only trusted the test after rebuilding the fixture and watching it fail
  for the right reason. Good answer to "how do you know your tests are
  actually testing anything" — with a concrete example, not just the
  principle.
- **Health checks: 503, not 200-with-a-flag.** `/health` and `/ready` are
  deliberately two different endpoints with two different failure
  behaviors, and the test suite encodes *why*: an orchestrator (Render, k8s,
  anything doing rolling deploys) needs a real non-2xx to route traffic away
  from a container that isn't ready, not a 200 it has to parse a body to
  distinguish from success. Directly answers "how do you think about
  liveness vs readiness."
- **Python default-argument binding bit the test suite, not just the code.**
  Several functions read config via a default parameter —
  `def load_documents(data_dir: Path = config.DATA_DIR)` — which is
  evaluated once, at import time. Patching `config.DATA_DIR` in a test
  afterward silently does nothing to that function; it's still holding
  whatever the value was the moment the module was first imported. Worth
  bringing up either as a debugging story or, if asked "what would you
  refactor," as a concrete answer: read config inside the function body,
  not as a default value.

**Time:** TODO

## A1 — AWS account, guardrails, CLI — 2026-10-05

**Setup:**

- **Account:** AWS "new experience" project **Try Catch Finally** — Google
  sign-in, Free plan, $100 credit, no card. Credits: $___, expire ____.
  A second, older AWS account exists under my email (no credits) — unused,
  nothing gets built there.
- **Region: `ap-southeast-2` (Sydney).** Only region used. Assigned by AWS:
  new-experience sign-ups from Asia-Pacific are placed in Sydney and the
  Region can't be changed without activating "advanced features". (Earlier
  plan said `ap-south-1` Mumbai — corrected.)
- **Admin identity:** IAM user `RAGproject` (AdministratorAccess). CLI auth
  via an access key stored only in `~/.aws/credentials` — never in the repo,
  `.env`, the image, or EC2. EC2 will use an IAM role instead (A2).
  `aws sts get-caller-identity` → `...:user/RAGproject`.
- **MFA: deferred.** Not yet enabled on the Google sign-in or on `RAGproject`.
  Open item — do it before A2 goes live (settings.aws.com → Security →
  Register device, and/or Google 2-Step Verification).
- **Budget:** TODO — $5 monthly cost budget, or note here if the Free plan
  blocks Budgets. The Free plan can't charge anyway; paid projects get a hard
  spend cap that pauses the project rather than billing past it.

**Findings:**

- **The console kept opening in Sydney, and that was the platform, not me.**
  I first assumed a wrong default and tried to force Mumbai. Then the menu
  showed *Projects / Team / Switch project* instead of *Security
  credentials*, which is how I worked out this is AWS's new project-based
  sign-up, where the Region is fixed per project. Lesson: check *which kind*
  of account you're in before following a tutorial written for classic
  accounts.
- **Free plan blocks some console services.** Unified Settings (default
  Region) returned "Upgrade to paid plan". Didn't upgrade — that would end
  the $0 cap. Region is enforced by the CLI default
  (`aws configure set region ap-southeast-2`) and by checking `region=` in
  the console URL before creating anything.
- **IAM user created without console access.** `RAGproject` was created with
  only programmatic access, so no password and no Console sign-in section.
  Fix if needed: `aws iam create-login-profile --user-name RAGproject ...`.
- **Credits are per account, not per user.** Seeing $100 under one login and
  $0 under another meant two different accounts, not a permissions issue.

**Time:** TODO

## A2 — Image to ECR, running on EC2 — 2026-10-06

**Goal:** push a versioned image to ECR and run it on EC2 with no AWS
credentials on the box.

**Setup:**

- **ECR:** repository `rag-app`; image pushed as
  `272037674852.dkr.ecr.ap-southeast-2.amazonaws.com/rag-app:v1`, built with
  `docker buildx build --platform linux/amd64 --provenance=false ... --push`.
  `--platform` because the Mac is ARM and EC2 is x86 — an ARM image dies there
  with `exec format error`. Tagged `v1`, never `:latest`, so rollback is
  always possible.
- **IAM role `rag-ec2-role`:** EC2 trust, one managed policy —
  `AmazonEC2ContainerRegistryReadOnly`. The box pulls from ECR on short-lived
  role credentials; no access keys ever on it. The console created the
  instance profile (the wrapper that lets EC2 use a role) silently; the CLI
  and Terraform make you create it explicitly.
- **EC2:** Ubuntu 24.04 x86, t3.small (2GB — app ~300MB plus Docker, OS and
  the boot-time ingest), launched from the console wizard in
  `ap-southeast-2`. User data installs `docker.io` and AWS CLI v2 and adds
  `ubuntu` to the docker group.
- **Security group:** 22 from my IP only, 8000 from anywhere — the API is
  public, the shell is not.
- **Secrets:** only `OPENAI_API_KEY`, `API_KEY`, `TAVILY_API_KEY` copied over
  with `scp` into `/opt/rag/.env` (chmod 600). Filtered with
  `sed 's/^ *//' .env | grep -E ...` — my local `.env` has a leading space on
  the Tavily line. Never in the image, never in git.
- **Run:** `docker run -d --restart unless-stopped -p 8000:8000 --env-file
  /opt/rag/.env -e CHROMA_DIR=/state/.chroma -e
  CHECKPOINT_DB=/state/checkpoint.db -v /opt/rag/state:/state ...:v1`

**Result:**

- `curl http://<public-ip>:8000/health` from my Mac → `200 {"status":"ok"}`.
- `/ready` → `{"status":"ready","documents":716}`.
- **Cold boot: 30s** (container start → `/ready`, timed with
  `time (until curl -sf localhost:8000/ready; do sleep 2; done)`).
- Warm restart (`docker restart rag`): ___s — TODO.
- **State:** the index and `checkpoint.db` live on a host volume, so they
  survive a container restart here. On Render there is no volume, so every
  redeploy re-ingests and conversation memory resets. That contrast is the
  answer to "how do you handle state?".
- **Chunk count differs from Render** (716 here vs 692 there) from the same
  Dockerfile — the corpus is fetched from arXiv at build time, so builds on
  different days can get different PDF versions. Second data point for the
  "pin arXiv versions" open item.

**What broke:**

1. `ssh ... Permission denied (publickey)`.
2. `curl` to port 8000 from my Mac timed out after 75s, while `curl
   localhost:8000/ready` on the instance worked.
3. A pasted block of AWS CLI commands stopped after the first one.

**How I diagnosed it:**

1. The server answered, so network and the port-22 rule were fine — it was
   authentication. `ls -l ~/.ssh/rag-key.pem` → no such file: the key was
   still in Downloads. When `-i` points at a missing file, ssh only warns and
   falls back to default keys, which the instance rejects.
2. Worked on the box → the app and the `-p 8000:8000` mapping were fine. A
   *timeout* rather than *connection refused* means packets were dropped, not
   that nothing was listening — on AWS that's the security group. It had no
   inbound rule for 8000 (missed in the wizard's network step).
3. `create-repository` printed long JSON, which the AWS CLI opened in a
   pager; quitting the pager discarded the rest of the paste.

**Fix:** moved the key to `~/.ssh/` + `chmod 400`; added inbound rule Custom
TCP 8000 from `0.0.0.0/0` (takes effect immediately, no reboot);
`--no-cli-pager` on CLI commands with long output.

**Loose ends:** instance is Name-tagged `rag-ec2-role` (same as the IAM
role — rename to `rag-app`); public IP changes on every stop/start (expected,
no Elastic IP); MFA still not enabled (from A1).

**Time:** TODO
