# Multi-agent conversation evaluation

Evaluating a conversational agent is harder than evaluating a single completion:
output is stochastic, quality is multi-dimensional, and there is no reference answer
to diff against. This is a working evaluation harness for that problem — a simulated
learner drives multi-turn conversations against the agent under test, a panel of
judges from different providers scores each transcript, and an orchestrator collapses
those scores into a pass/fail decision plus a queue of cases for a human.

```bash
python -m conveval evaluate                 # the main entry point: run + score + route
python -m conveval pipeline                 # the architecture and model roles
python -m conveval run                      # cached fixtures: instant, offline, free
mlflow ui --backend-store-uri sqlite:///mlflow.db   # NOT plain `mlflow ui`
```

**[DOCS.md](DOCS.md)** is the full writeup: pipeline diagram, the design decisions worth
asking about, and a tour of what each MLflow tab shows.

---

## The problem this solves

One evaluation sweep produces scores across four nested dimensions:

```
scenario  x  run  x  criterion  x  judge  ->  verdict
```

Almost all of the design is in **what order you collapse those four, and with which
function at each step**. Collapse in the wrong order, or with one function for
everything, and you get a healthy-looking number sitting on top of a broken system.

## Architecture

```
                        +------------------------------+
                        |  Orchestrator                |
                        |  fan out: scenarios x runs   |
                        +--------------+---------------+
                                       |
                        +--------------v---------------+
                        |  Simulated-learner agent     |  persona + goal + seed
                        +--------------+---------------+
                                       |  multi-turn, LangGraph loop
                        +--------------v---------------+
                        |  Coaching agent (under test) |  grounded on a context
                        +--------------+---------------+
                                       |  transcript
              +------------------------+------------------------+
              v                        v                        v
        claude-sonnet-4.5     gemini-2.5-flash       llama-3.3-70b
        score + evidence      score + evidence       score + evidence
              +------------------------+------------------------+
                                       v
                     STEP 1   collapse JUDGES     -> consensus + dispersion
                                       v
                     STEP 2   collapse RUNS       -> scenario result
                                       v
                     STEP 3   collapse SCENARIOS  -> scorecard + gates
                                       v
                     scorecard | gates | human review queue
```

### Why a panel, and why across providers

Two separate reasons, and they need different fixes.

**Variance.** A single judge on a single sample is noisy; scores swing between runs
on identical input. More judges reduce that.

**Bias.** A model rates output from its own family higher — self-preference bias. The
agent under test runs on **OpenAI**; the three judges are **Anthropic, Google and
Meta**. No judge shares a family with the thing it is scoring.

Every role routes through OpenRouter, so this is one key rather than four accounts,
and each role is env-overridable:

```bash
CONVEVAL_JUDGE_3=mistralai/mistral-small-3.2-24b-instruct python -m conveval run --rejudge
```

Because judges can be swapped that freely, independence is **asserted, not assumed**:
`judge_families_are_independent()` is checked in the report and in the test suite, so
reconfiguring the panel cannot silently reintroduce the bias it exists to avoid.

### Step 1 — collapse judges first

Judge disagreement is **measurement noise**. It has to be resolved or surfaced before
it propagates into any higher-level average, so it collapses first.

- **binary** criteria: majority vote
- **ordinal** criteria: **median, not mean.** With judges scoring 4, 4 and 1, the mean
  is 3.0 — a score nobody gave, decided by one outlier. The median is 4, and the
  dispersion separately records that someone dissented.

Dispersion is then the **routing signal**: a non-unanimous panel means the criterion is
ambiguous or the transcript is genuinely borderline, and that is what a human should
look at. This is the mechanism behind reduced manual review — reviewers stop reading
everything and read only the contested minority.

### Step 2 — collapse runs, with a different function per criterion

This is the part most eval suites get wrong by using one function everywhere:

| criterion | type | across runs | why |
|---|---|---|---|
| faithfulness | binary | **pass rate** — "2/3 runs" | actionable in a way that "0.67" is not |
| in scenario | binary | pass rate | same |
| pedagogy | ordinal 1-5 | **mean + interval** | quality; the interval stops noise reading as improvement |

**Binary where the question is factual, ordinal only where the judgement is graded.**
The reason is inter-rater agreement: a binary has one obvious reading, while a 1-5
scale invites each judge to anchor differently. This suite's own results bear that
out — every unanimous criterion is binary, and the one the panel splits on is the
ordinal. Pedagogy stays *despite* being the noisiest, because it is the only criterion
that exercises median consensus, mean-with-interval aggregation and dispersion routing.

`any_occurrence` still exists in the aggregation layer for safety-shaped criteria; it
simply has no custom criterion using it now that safety is a built-in.

Averaging a safety criterion lets one catastrophic run hide behind two good ones: ten
runs containing one unsafe reply would report 90% and read as broadly fine. The
`any_failure` rule is what stops that, and it is why the aggregation function is a
property of the criterion rather than a global setting.

### Step 3 — collapse scenarios, but never into one number

A mean across scenarios hides one broken scenario behind the healthy ones, which is
precisely the failure the suite exists to catch. Output is a **scenario x criterion
matrix** plus **hard gates**:

- no safety failure anywhere
- correctness floor, evaluated **per scenario** rather than averaged across them
- **panel agreement** above threshold — this gates the *instrument*, not the system.
  If judges cannot agree, every other number in the report is noise.
- every accusation's cited evidence must be verifiable
- no regression against the committed baseline — in CI the useful question is "did
  this change make anything worse", not "what is the absolute score"

---

## What the panel actually caught

The strongest evidence for a multi-judge panel came from a bug it found in **this
harness**, not in the agent.

The `happy_path` scenario kept coming back **contested**, which made no sense for a
case meant to be clean. Two judges passed it; `claude-sonnet-4.5` failed it on
"stayed in scenario", with this rationale:

> *The coach engages with the learner's invented scenario about interruptions and
> dismissals rather than redirecting to the actual scenario facts about Sam missing
> three sprint handovers.*

It was right, and the other two judges missed it. The scenario facts said Sam had
missed three sprint handovers; the transcript never mentioned handovers once. The
learner had invented a completely different problem and the coach had followed.

**Root cause:** the simulated learner was given a persona and a goal but never the
scenario facts, so it had nothing to talk about and made something up. The
conversation still read as perfectly plausible, which is exactly why a single judge -
or a human skimming - would have passed it.

Two fixes followed:

1. **Scenario facts are now given to both sides.** The learner receives the situation
   it is bringing; the coach additionally receives the framework it must follow. The
   framework is deliberately *not* shared with the learner, or it would coach itself.
2. **The faithfulness rubric was disambiguated.** The same judge also failed the coach
   for saying the words "the SBI framework" out loud, reading "asserts a named
   framework not in the context" as "must not name the framework". The framework *was*
   in its context. The rubric now says explicitly that the criterion is about
   INVENTING information, not about which instructions the coach says aloud.

After both fixes the three scenarios do exactly what they are named for: `happy_path`
passes unanimously, `unfaithful` fails unanimously, and only `disagreement` splits the
panel. Both fixes have regression tests.

The general lesson is the useful one: **panel disagreement is a signal about the
measurement, and the measurement includes the harness.** A single judge would have
returned a clean pass and the broken scenario would have gone unnoticed.

---

## Verifying the judges in code

Every judge must return the **verbatim span** it is accusing the agent of. `verify.py`
then checks in plain Python that the span occurs in what the agent actually said. A
judge citing text that does not exist is hallucinating, and that is catchable for
free — no second model, no extra tokens.

**This caught a real judge failure during development.** A judge failed a transcript
on faithfulness while quoting *"Use the SBI structure: Situation, Behaviour, Impact."*
— a line from the **coaching context given to the agent**, not from anything the agent
had said. The other two judges passed it. That transcript was therefore both a 2-1
split routed to a human and a flagged unverifiable accusation, caught by string
matching rather than by another model.

The hard part is **false positives**, and getting this wrong is how such a check ends
up switched off. Three real failure modes, all observed on the first run:

1. Judges quote the transcript **as presented to them**, including the `COACH:` speaker
   label, which is not part of the agent's own text.
2. Judges cite several spans joined by `;` or newlines, not one contiguous quote.
3. Models reformat: curly apostrophes, collapsed whitespace, `...` for an elision.

Beyond that, evidence is only demanded where it is *meaningful*. A specific accusation
("it invented this statistic") has a locatable span. A holistic judgement ("it never
built on the learner's answer") does not, and demanding one manufactures failures —
which it did, on the first run, before `requires_evidence` was introduced.

---

## Three scenarios, each demonstrating one thing

An evaluation demo where everything passes proves nothing: the scorecard is a number
and the machinery is invisible. Each scenario exercises a different part of the
pipeline (`scenarios.py`).

| scenario | seeded behaviour | demonstrates |
|---|---|---|
| `happy_path` | none | a clean run; the panel agrees; every gate green |
| `unfaithful` | invents a named study and a salary band | a clean catch: faithfulness fails and the panel agrees |
| `disagreement` | warm but purely reactive coaching | **dispersion routing** — the panel splits and a human is asked to look |

The third is the interesting one. The first two show the pipeline working; the third
shows what it does when *the measurement itself* is uncertain, which is the case a
single-judge setup cannot even detect.

Runs default to **1 per scenario**, so the demo is three legible transcripts.
`--runs 3` fans out, which is what pass-rate and any-occurrence aggregation exist for:
against a stochastic system, one sample per scenario tells you very little.

---

## MLflow is the interface

Built on **`mlflow.genai.evaluate()`**, not the generic tracking API. The difference is
the whole point:

| generic tracking | `genai.evaluate` |
|---|---|
| a tree of separate runs (suite / scenario / judge) | **one run, one ROW per scenario** |
| browsable, but nothing connects | the trace, the transcript and every judge's assessment hang off the same row |

Each criterion is a `@scorer` returning a **`list[Feedback]`**: one assessment per
judge, attributed with `AssessmentSource(LLM_JUDGE, source_id=<model>)`, plus a
consensus assessment carrying the aggregation result. So the Traces table shows
columns like:

```
Request                              faithfulness  __claude-sonnet  __gemini  __llama   Safety
{"scenario_id": "unfaithful"}        fail          fail             fail      fail      Pass
{"scenario_id": "happy_path"}        pass          fail             pass      pass      Pass
```

The panel is visible as data, not buried in an artifact. An aggregate that hides a
2-1 split is exactly what this project argues against, so the split is logged.

Each judge's assessment carries, in metadata: the **verbatim span it cited** and
whether that span was **verified** to occur in the agent's output.

### Human review, without the Review App

MLflow's Review App is **Databricks-only** (`get_review_app` says so explicitly). In
OSS the equivalent is tagging: contested traces are tagged after the run, so a reviewer
filters the Traces table on

```
tags.contested = 'true'
```

and sees only the rows where the automated scores are unreliable. That is all a
reviewer needs here - they have to *see* disagreement, not record a decision.

Tagging happens as a post-pass, not inside the scorer: a scorer runs outside any trace
context, so `mlflow.update_current_trace` there is a silent no-op.

### Built-in scorers vs custom judges

Safety is **not** hand-rolled. MLflow ships a validated `Safety` judge, so this uses it.
Custom judges are reserved for what the domain actually needs - faithfulness to a
supplied coaching context, and pedagogical progression - neither of which has a
built-in equivalent.

Two traps worth knowing:

- `ConversationalSafety` is a **session-level** scorer and needs traces carrying
  session IDs. Mixing one into a row-level evaluation fails outright. The row-level
  `Safety` is the one that fits here.
- Built-in judges resolve `openai:/...` through the OpenAI SDK, which reads
  `OPENAI_API_KEY`. This project only has an OpenRouter key, so every built-in failed
  with a missing-key error until the SDK's base URL was pointed at OpenRouter.

> **Start the UI with the backend URI.** Plain `mlflow ui` reads `./mlruns` and shows
> an empty experiment list. Runs live in SQLite because MLflow 3.x puts the filesystem
> backend in maintenance mode and refuses it:
>
> ```bash
> mlflow ui --backend-store-uri sqlite:///mlflow.db
> ```

---

## Running it

| command | what it does | cost |
|---|---|---|
| `python -m conveval run` | cached fixtures | free, offline, instant |
| `python -m conveval run --rejudge` | recorded transcripts, live judge panel | ~27 calls |
| `python -m conveval run --regenerate` | rebuild conversations and verdicts | ~120 calls |
| `python -m conveval explain unfaithful#run1` | walk one transcript through the collapse | free |
| `python -m conveval run --runs 3` | fan out 3 runs per scenario | 3x the calls |
| `python -m conveval pipeline` | architecture + model roles + independence check | free |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db   # NOT plain `mlflow ui`` | browse and drill into every run | free |

Cached fixtures are **real recorded model output**, captured once — the demo shows
genuine cross-provider judging, it does not simulate it.

Needs `OPENROUTER_API_KEY` in `.env` for the live modes. One key covers every role.

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

---

## Where LangGraph is used, and where it is not

LangGraph drives the learner/coach turn loop in `conversation.py` and nothing else.
That loop is genuinely cyclic with a termination condition, which is the shape graphs
are for. The fan-out across scenarios and runs is plain concurrency in `runner.py`,
because fan-out/fan-in is not a state machine and wrapping it in a graph would be
ceremony.

## Layout

```
conveval/
  models.py        the four dimensions, as types
  scenarios.py     scenarios, rubric, seeded flaws, baseline
  conversation.py  LangGraph learner <-> coach loop
  judges.py        the panel; one structured call per judge per transcript
  verify.py        deterministic evidence verification
  aggregate.py     THE CORE: judges -> runs -> scenarios
  runner.py        orchestration, caching, modes
  report.py        CLI rendering
  agent.py         the system under test, behind a black-box Protocol
  tracking.py      MLflow: the nested run tree
tests/             aggregation logic, one test per design decision
```

The pipeline is headless: it writes `results.json` and logs to MLflow. There is no UI
code to maintain and nothing UI-shaped can break a demo or a CI run.

**The agent under test is reached only through `AgentUnderTest.respond()`.** The harness
never learns how it is implemented, so the same suite can point at a prompted model, a
LangGraph app, or an HTTP endpoint fronting a service in another language. Deliberately
NOT built on the OpenAI Assistants API, which sunsets 2026-08-26; coupling to a vendor
runtime is the exact mistake this interface avoids.

## What I would add next

- **Calibration against human labels.** The judges are a proxy and a proxy should be
  validated. The UI already records adjudications to `labels.json`; the missing piece
  is measuring judge-vs-human agreement over them, which is a different and more
  important question than judge-vs-judge.
- **A formal agreement statistic** (Krippendorff's alpha) rather than the simple
  agreement fraction used here.
- **A judge cascade** — a cheap model scores everything, escalating to the full panel
  only near a gate threshold. Judges x runs x scenarios multiplies cost quickly.
- **More binary criteria, fewer scales.** Agreement was visibly higher on the binary
  criteria than on the one ordinal criterion.
