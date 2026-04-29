# StochPy

> Composable deterministic and stochastic functions for Python — where LLMs, humans, and algorithms work as equals.

---

## Why

Every programming language treats functions as deterministic — same input, same output, always.

But modern software increasingly calls LLMs, which are fundamentally different: same input, *probabilistic* output. The standard response has been to bolt LLM calls onto existing languages as a library concern — you call the model, parse the string, handle failures, validate output, retry, escalate. All of that is your problem.

StochPy takes a different position: **stochastic functions should be first-class citizens of the language**, with the same composability, type safety, and error handling as deterministic ones.

The design is inspired by how neural networks work — reliability emerging from the composition of probabilistic units. No single node needs to be exact for the whole to be trustworthy.

---

## What We Built

StochPy is a Python library extension with 8 primitives that compose freely:

```
@interface     typed output contracts — the boundary between stochastic and deterministic
@context       a single stochastic LLM call, eval-gated, contract-enforced
@agent         a self-directing loop — runs until an objective is satisfied
ModelRegistry  runtime model selection — cost, quality, or latency optimised
PromptRegistry YAML-based prompt store — versioned, composable, human-editable
escalate()     native human-in-the-loop — execution suspends, waits, resumes
parallel()     typed concurrent execution with partial failure handling
MockModel      test harness — full pipeline without calling a real model
```

Flow control stays in plain Python. No graph DSL, no new syntax to learn.

---

## Core Concepts

### Interfaces — typed output contracts

The boundary between the stochastic and deterministic worlds. A model can be fuzzy internally — but what it returns must satisfy a declared shape.

```python
from pydantic import BaseModel
from stochpy import interface

@interface
class SpamCheck(BaseModel):
    is_spam:    bool
    confidence: float
    reason:     str
```

If the model output doesn't match → `ContractException`. Downstream code always receives a typed, validated object.

### @context — stochastic function

```python
from stochpy import context, ModelRegistry

registry = ModelRegistry(
    quality = "anthropic/claude-sonnet-4-6",
    fast    = "openai/gpt-4o-mini",
)

@context(
    model     = registry.quality,
    prompt    = prompts.load("app.spam.check"),
    eval      = [
        lambda out: 0.0 <= out.confidence <= 1.0,
        lambda out: isinstance(out.is_spam, bool),
    ],
    threshold = 0.85,
    retries   = 2,
)
def check_spam(sender: str, body: str) -> SpamCheck:
    ...
```

The function body is never called. The decorator is the engine — it renders the prompt, calls the model, parses the output into `SpamCheck`, runs eval, and retries on failure.

### @agent — self-directing loop

An agent doesn't follow a fixed pipeline. It observes state, decides which tool to call next, executes it, and loops until its objective is satisfied — or raises an exception.

```python
from stochpy import agent

@agent(
    tools     = [check_spam, identify_customer, search_faq],
    objective = [lambda out: out.resolved == True],
    model     = registry.quality,
    max_steps = 10,
    timeout   = 120.0,
)
def handle_complaint(email: str) -> ComplaintResolution:
    ...
```

Agents and contexts are interchangeable as tools — agents can call contexts, contexts can be tools inside agents.

### ModelRegistry — auto model selection

```python
registry = ModelRegistry(
    fast     = "openai/gpt-4o-mini",
    balanced = "openai/gpt-4o",
    quality  = "anthropic/claude-sonnet-4-6",
    tracking = True,                            # learns over time
)

# fixed
@context(model=registry.quality, ...)

# runtime picks cheapest that passes eval
@context(model=registry.auto_cost, ...)

# runtime picks highest scoring for this context
@context(model=registry.auto_quality, ...)
```

With `tracking=True`, the registry records eval pass/fail per context per model. Models that fall below 60% success rate on a context are skipped in auto-selection. The language gets smarter about model selection without changing code.

### PromptRegistry — prompts out of code

Prompts live in YAML, not source files. Code only holds a key.

```python
@context(
    model  = registry.quality,
    prompt = prompts.load("finance.pl_graph.analyze"),
    ...
)
def analyze_pl_graph(graph_json: str, intent: ParsedIntent) -> PLGraph:
    ...
```

```yaml
# prompts/finance/pl_graph.yaml
analyze:
  version: 2.1
  system: |
    You are a financial analysis assistant.
  user: |
    Analyze this P&L graph.
    Graph:   {{ graph_json }}
    Intent:  {{ intent.intent }}
    Metrics: {{ intent.metrics }}
  partials:
    - shared.time_filter        # reusable fragments
  examples:
    - input:  {intent: "variance_analysis", segments: ["Segment A"]}
      output: {relevant: ["seg_ebitda", "seg_revenue"]}
```

Prompt changes don't require code deploys. Non-technical domain experts can tune prompts directly. Full git versioning. A/B test by loading a different version.

### Human-in-the-loop — native construct

Human is a first-class executor — not a webhook bolted on the side.

```python
from stochpy import escalate, ApprovalGate, role

# escalate on low confidence — execution suspends
except ConfidenceException:
    result = await escalate(
        to      = role.senior_assessor,
        message = "Low confidence classification — please review",
        timeout = "4h",
    )

# approval gate — pipeline pauses until CFO approves
async with ApprovalGate(required=role.cfo, timeout="24h",
                        message="Please review Q3 report before distribution"):
    distribute(report)
```

The human response is validated against the same interface contract as the model output. Whether a model or a human resolves a step — downstream code sees the same typed object.

### Plain Python for flow control

No graph DSL. Pipelines are just Python:

```python
from concurrent.futures import ThreadPoolExecutor
from stochpy import parallel

def handle_complaint(email: Email) -> ComplaintResolution:

    # parallel stochastic calls — just threads
    result = parallel(
        lambda: check_spam(email),
        lambda: check_language(email),
    )

    if result.failed:
        log_partial_failure(result.failed)

    spam, language = result.succeeded

    if spam.is_spam:
        return ComplaintResolution(method="spam_rejected")

    customer = identify_customer(email)       # stochastic
    if not customer.found:
        return await escalate(role.support, email)

    for attempt in range(3):                  # retry loop — plain Python
        faq = search_faq(email, customer)
        if faq.answered:
            return ComplaintResolution(method="faq", response=faq.answer)

    return await escalate(role.support_agent, email, customer)
```

### Testing

```python
from stochpy.testing import MockModel

mock = MockModel()
mock.when("check_spam", '{"is_spam": false, "confidence": 0.95, "reason": "ok"}')

with mock.patch():
    result = check_spam(sender="alice@example.com", body="Hello")
    assert result.is_spam    == False
    assert isinstance(result, SpamCheck)      # interface still validated

mock.assert_called("check_spam", times=1)
```

The full eval, contract validation, and exception pipeline runs — only the actual model call is bypassed.

---

## Exception Hierarchy

```
StochPyException
  ├── ContractException          output didn't match interface
  │     ├── EvalAssertException  assert lambda returned False
  │     ├── EvalScoreException   score below threshold
  │     └── ConfidenceException  model confidence too low
  ├── ModelException             provider error
  │     ├── ModelTimeoutException
  │     └── ModelUnavailableException
  ├── ObjectiveException         agent couldn't satisfy objective
  │     ├── MaxStepsException
  │     └── AgentTimeoutException
  ├── EscalationException
  │     └── EscalationTimeoutException
  └── ApprovalRejectedException
```

Stochastic exceptions plug into the same `try/except` machinery as classical ones. Retrying a stochastic failure resamples the distribution — unlike classical retry, this actually makes sense.

---

## Executor Spectrum

One of the core ideas in StochPy: the thing that resolves a computation is a runtime decision, not a code decision.

```
def              pure deterministic function
@context         model resolves once, contract-gated
@agent           model loop until objective satisfied
escalate()       human resolves
ApprovalGate     human approves, pipeline resumes
[auto, human]    model first, human fallback
federated        panel of models, consensus merge
```

Model providers become commodity compute — you declare what the output must satisfy, the runtime decides who produces it.

---

## Reference Use Case — CFO Assist

The library was designed and validated against a real-world scenario: an AI assistant for a CFO that receives natural language financial analysis requests, walks a P&L graph to identify relevant data nodes, generates DAX queries against a data warehouse, runs them in parallel, and consolidates the results into a structured financial report.

The flow demonstrates every primitive:
- `@context` for intent parsing, P&L graph analysis, DAX generation, report building
- `parallel()` for concurrent data warehouse queries
- Plain Python controllers for routing and retry logic
- `escalate()` for low-confidence results and data failures
- `ModelRegistry` auto-selection across quality tiers

---

## Project Structure

```
stochpy/
├── stochpy/
│   ├── exceptions.py        exception hierarchy
│   ├── interface.py         @interface decorator
│   ├── prompt_registry.py   YAML loader, partials, Jinja2 rendering
│   ├── registry.py          ModelRegistry, auto-selection, tracking
│   ├── context.py           @context decorator — core engine
│   ├── parallel.py          ThreadPoolExecutor wrapper
│   ├── human.py             escalate(), ApprovalGate
│   ├── agent.py             agent loop
│   ├── testing.py           MockModel test harness
│   └── __init__.py
├── prompts/                 YAML prompt files (user-owned)
└── tests/                   44 tests
```

---

## Installation

```bash
pip install stochpy
```

Requires Python 3.11+. Model calls use [LiteLLM](https://github.com/BerriAI/litellm) — set the relevant API key for your provider:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

---

## Status

`v0.1.0` — core primitives implemented and tested. Planned next:

- [ ] CFO Assist reference implementation
- [ ] Async `@context` and `@agent` variants  
- [ ] `FederatedSelector` — ensemble model execution
- [ ] Slack / Teams transport for `escalate()`
- [ ] CLI for prompt registry management
- [ ] PyPI publish
