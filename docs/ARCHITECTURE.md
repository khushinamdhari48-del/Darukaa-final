# Architecture

## The design decision everything else follows from

The brief rules out "generic LLM-only solutions". This system takes that
literally: **the language model cannot produce a recommendation, an effect size,
a citation, a confidence level or a time horizon.** Those are all computed by a
deterministic engine over a curated knowledge base. The model does two bounded
jobs — parsing free text into a typed schema, and phrasing the engine's output —
and both have a working non-LLM fallback.

The practical consequence is that `ANTHROPIC_API_KEY` is optional. With no key,
the system still diagnoses, retrieves, reasons, cites, and answers in full
markdown. That is not a degraded mode bolted on afterwards; the deterministic
path is the reference implementation and the LLM is the optional polish.

It also means the system is **testable**: the same site profile always yields the
same assessment, so the 116 tests in `tests/` assert on scientific reasoning
rather than on the mood of a sampler.

```
                       ┌──────────────────────────────────────────┐
  free text ──────────►│  dialogue/extract.py                     │
  "SOC 0.3%, 420mm,    │  regex + unit conversion + negation      │
   monoculture wheat"  │  (LLM fills gaps only, never overrides)  │
                       └───────────────────┬──────────────────────┘
                                           │  SiteProfile patch
  structured JSON ─────────────────────────┤
  (POST /assess)                           ▼
                       ┌──────────────────────────────────────────┐
                       │  dialogue/state.py                       │
                       │  MEMORY IS THE PROFILE, NOT THE          │
                       │  TRANSCRIPT. Turn 7 reasons over a       │
                       │  number from turn 2; a correction        │
                       │  overwrites the slot.                    │
                       └───────────────────┬──────────────────────┘
                                           ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │  reasoning/engine.py — deterministic, no network, no LLM             │
   │                                                                       │
   │  1. metrics.py    raw values ──► banded condition scores ──► 5        │
   │                   composite indices (each with its own coverage)      │
   │  2. diagnostics.py 27 auditable rules + limiting-factor analysis      │
   │                   traced backward through the causal graph            │
   │  3. recommend.py  candidates ──► contraindication filter ──►          │
   │                   evidence retrieval ──► effect composition ──►       │
   │                   confidence ──► priority ──► set-level interactions  │
   │  4. clarify.py    counterfactual simulation to rank what to ask next  │
   └──────────┬──────────────────────────────────┬─────────────────────────┘
              │                                  │
              ▼                                  ▼
   ┌────────────────────────┐        ┌──────────────────────────────┐
   │ knowledge/retriever.py │        │ datasets/causal_graph.yaml   │
   │ hybrid, site-conditioned│       │ 45 signed, weighted, lagged  │
   │  dense (0.45)          │        │ edges — every one sourced to │
   │  BM25   (0.30)         │        │ specific evidence cards      │
   │  site-condition (0.25) │        └──────────────────────────────┘
   └────────┬───────────────┘
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 73 evidence cards / 71 distinct citations (all DOI-verified) │
   │ 41 interventions · 73 metric bands · 22 richness baselines   │
   └──────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────┐
                       │  dialogue/render.py                      │
                       │  complete markdown, always               │
                       │  LLM (optional) paraphrases it and the   │
                       │  verbatim engine output is kept below    │
                       └──────────────────────────────────────────┘
```

---

## 1. The knowledge layer

Three kinds of knowledge, deliberately separated because they are used
differently.

### Evidence cards — `knowledge/corpus/*.yaml`

73 cards, each one atomic, citable and quantified. A card is not a text chunk; it
is a structured finding:

| Field | Purpose |
| --- | --- |
| `claim` | the finding, with numbers |
| `mechanism` | **why** it is true — the causal story, not a restatement |
| `effects[]` | machine-readable `{metric, direction, magnitude, time_to_effect}` so the engine can *compose* effects rather than only quote them |
| `applies_when[]` | predicates over site fields — the applicability envelope |
| `caveats[]` | what breaks it; these become the recommendation's risks |
| `citation` | authors, year, venue, DOI/URL, and an evidence *type* that drives the confidence calculus |
| `interventions[]` | which actions this card can support |

The `mechanism` field is what separates this from a document store. A retrieved
chunk can tell you cover crops raise soil carbon; a mechanism tells you *why*
legume mixes sit at the top of the range (nitrogen lowers residue C:N, raising
microbial carbon-use efficiency, so more carbon is stabilised as necromass) —
and that is the part that transfers to a site the study never covered.

`applies_when` is what makes the knowledge *conditional*. Evidence is not
universally true, and a system that retrieves a Sahel parkland study for a
temperate heath is wrong even though its search worked.

### Structured reference data — `knowledge/datasets/*.csv`

- **`metric_bands.csv`** (73 rows) — the interpretation layer. Turns 0.3% SOC
  into `critical` with a sourced explanation of *why* that band matters. This is
  where "0.3" becomes "below the ~1% threshold at which aggregate stability,
  water-holding capacity and soil faunal habitat decline together".
- **`richness_baselines.csv`** (22 rows) — expected species richness by land use
  × climate zone × taxon. Lets the engine judge an observed count against a
  **land-use-matched** baseline rather than against pristine reference, which is
  the only defensible comparison for a working farm.
- **`zone_reference.csv`** — agro-ecological zones, used to infer climate zone
  from rainfall (marked `inferred`, never asserted as measured).

### The causal graph — `knowledge/datasets/causal_graph.yaml`

45 edges, each carrying `sign`, `strength`, `lag_years`, a `mechanism` and the
evidence card ids that justify it. The loader **refuses to start** if any edge is
unsourced or cites a card that does not exist.

Three edge features do real work:

- **`sign: non_monotonic`** with an `optimum`. Groundwater recharge peaks at
  intermediate tree cover: below the optimum trees help, above it they hurt.
  Without a site value the engine reports `context_dependent` rather than
  guessing — and that becomes a clarifying question.
- **`saturating_at`** — above this level of the *source* metric, further movement
  buys much less. This is what stops the engine promising a 0.3%-SOC gain to a
  peat soil.
- **`threshold`** — below this level the edge is inert. Nitrogen only suppresses
  plant richness once it has stopped being the limiting nutrient; below that,
  adding nitrogen is *good* for the site.

### Interventions — `knowledge/datasets/interventions.yaml`

41 actions with `targets`, `addresses` (which diagnoses), `requires`,
`contraindications`, cost, reversibility, establishment period, a monitoring
plan, declared synergies/conflicts, and a `novelty` flag. An intervention with no
supporting evidence card **cannot load** — so nothing is recommendable that
cannot be cited.

---

## 2. Retrieval — `knowledge/retriever.py`

Three signals are fused per card:

```
score = 0.45 · dense(query, card)        # semantic match, min-max normalised
      + 0.30 · bm25(query, card)         # rare technical terms hashing dilutes
      + 0.25 · condition_match(site, card)   # ← the part generic RAG lacks
      × 0.15 if any applies_when predicate is VIOLATED
```

**Why hybrid.** The decisive tokens in this domain are rare: *sodicity*,
*hydroperiod*, *zai*, *Faidherbia*, *neonicotinoid*. Feature hashing dilutes
exactly those, so BM25 runs alongside, with the curated `tags` field weighted up
(standard IR field boosting). Measured on a 12-query expert-labelled set:
**recall@1 = 67%, recall@3 = 92%** (`tests/test_retrieval.py::test_relevance_baseline`
holds this as a regression floor).

**Why the third signal matters.** "Biodiversity is declining on my land" is
roughly equally similar to a Sahel agroforestry card and a temperate
nitrogen-deposition card. Only the site profile can break the tie. Condition
matching is three-valued — satisfied / violated / **unknown** — and unknown gets
half credit, because absence of information about a site should not count as
evidence against a card.

**Violated evidence is not silently dropped.** It is penalised, partitioned, and
retrievable on request, because *"this popular measure does not apply to you,
because your rainfall is below 350 mm"* is frequently the most useful output.
`find_contradicted()` exists specifically for that, and `/knowledge/search` with
`include_violated: true` exposes it.

**Backends are swappable.** Default is a dependency-free persistent numpy store
with exact cosine and a deterministic hashed-TF-IDF embedder — at 73 cards, exact
search is both faster and more accurate than ANN, so the portable choice is also
the correct engineering one. `BIOAI_VECTOR_BACKEND=chroma` and
`BIOAI_EMBEDDER=minilm` swap in ChromaDB and MiniLM; both degrade with a warning
rather than crashing if the optional dependency is absent.

---

## 3. Reasoning

### Normalisation and composite indices — `reasoning/metrics.py`

Raw values become 0–1 condition scores via the band table, so pH 4.9 and SOC
0.3% become commensurable. Five composites follow: soil health, water security,
habitat integrity, human pressure, biodiversity state.

Aggregation is **minimum-sensitive**: `0.7 · weighted_mean + 0.3 · min`. A plain
mean would let five good components mask one critical one, which is precisely the
failure mode in environmental assessment (Liebig's law of the minimum). Every
index also reports its own **coverage**, and an index computed from one of six
components is flagged as indicative rather than measured.

### Diagnosis before prescription — `reasoning/diagnostics.py`

The engine never jumps from symptom to intervention. Two mechanisms produce a
ranked set of diagnoses:

1. **27 auditable rules**, each naming the metrics it implicates and the
   interaction that makes the factor limiting.
2. **Limiting-factor analysis** — for each degraded outcome metric, the causal
   graph is traced *backward* and each upstream metric is scored by
   `edge_strength × how_degraded_it_actually_is`. An influential but healthy
   upstream metric explains nothing; one that is both influential and degraded is
   the limiting factor. This can surface a cause no rule anticipated.

### Recommendation construction — `reasoning/recommend.py`

```
diagnoses
  ─► candidates          an intervention must address a fired diagnosis
  ─► hard filter         contraindications drop it, WITH a stated reason
  ─► evidence            site-conditioned retrieval per candidate; the query is
                         built from the intervention + its diagnoses, so
                         retrieval is steered by the reasoning, not just the
                         user's wording
  ─► effect composition  card effects = PRIMARY (cited, quantified)
                         graph propagation = SECONDARY (labelled as modelled)
  ─► confidence          evidence_type weight × site-condition match
                         × corroboration, CAPPED by data completeness
  ─► priority            3.0·severity + 2.0·leverage + 1.6·breadth
                         + 1.5·evidence + 0.7·novelty − cost − slowness
  ─► set-level           synergies, conflicts, sequencing constraints
```

Four parts of this are worth calling out:

**Leverage** measures how degraded the metrics an intervention *moves* currently
are. A measure with excellent literature support that improves an
already-healthy metric scores low. This is what stops the engine recommending
cover crops to a 4%-SOC soil — the single most common way naive recommenders
become confidently useless.

**Primary vs secondary effects are never blended.** A cited, quantified effect
from a study and a model-propagated consequence are different epistemic objects.
Secondary effects carry their pathway (`soil moisture -> +soil organic carbon ->
+earthworm abundance`) and the confidence block explicitly notes how many of the
impacts are modelled rather than measured.

**Confidence is capped by data completeness.** Strong literature on a
barely-described site does not justify a confident recommendation. The cap is
`0.45 + 0.5·min(completeness/0.6, 1)`.

**Exclusions are output, not silence.** Rejected candidates are reported as
"Ruled out for this site: *X* (requires rainfall ≥ 400 mm)".

### What to ask next — `dialogue/clarify.py`

Rather than asking for whatever field is empty in a fixed order, the engine
estimates the **decision value** of each unknown field by counterfactual
simulation:

```
for each unknown field:
    for each plausible probe value (low band / high band, worst / best category):
        re-run diagnosis + candidate generation with that value imputed
        measure how far the recommendation set moved
gain = 0.55·mean_divergence + 0.25·spread + 0.20·structural_prior
```

The **spread** term is the real information signal: if the low case and the high
case lead to the same advice, the answer decides nothing and the question is not
worth a turn. The structural prior (graph centrality, how many interventions gate
on the field, whether it sits on a non-monotonic edge) covers fields that matter
but whose probe values happen not to flip a rule.

This runs entirely on the deterministic layer — no LLM call, no retrieval — so it
costs ~50 ms and can run on every turn.

---

## 4. Dialogue

`dialogue/state.py` keeps the cumulative `SiteProfile` as the primary state, not
the transcript. Consequences:

- Turn 7 reasons over a number given in turn 2 without re-reading anything.
- "Actually, the pH is 6.4" **overwrites the slot** rather than leaving two
  contradictory claims in a context window.
- An inferred value never overwrites a user-stated one (`provenance` tracks
  `user` / `inferred` / `regional_default`, and the renderer labels them).

`dialogue/orchestrator.py` holds the ask-or-advise **policy**, which is
deliberately not a hard gate:

- ask below 18% completeness, or when nothing is recommendable and the top
  question has real information gain;
- but never more than twice in a row, so the system cannot get stuck
  interrogating a user who does not have the numbers;
- immediately stop asking on "just tell me", "I don't know", "assume";
- and even when advising, outstanding questions are carried at the end.

---

## 5. Interfaces

| Surface | Entry point |
| --- | --- |
| REST API + OpenAPI | `api/app.py` → `/chat`, `/assess`, `/knowledge/*`, `/session/*`, `/healthz` |
| Terminal client | `cli.py` → `chat`, `assess`, `demo`, `search`, `stats` |
| Web client | `web/index.html` — single file, no framework, no build step |

`/knowledge/search`, `/knowledge/graph` and `/knowledge/interventions` exist so
the knowledge layer can be inspected independently of the reasoning that consumes
it — the retrieval pipeline is auditable from outside, not just described in a
README.

---

## 6. Failure behaviour

| Failure | Behaviour |
| --- | --- |
| No API key | Full deterministic operation; `/healthz` states it plainly |
| LLM call fails or returns bad JSON | Logged, falls back to the deterministic path |
| LLM returns a hallucinated field | Validated field-by-field against the schema and discarded |
| Malformed evidence card | Load fails at **startup**, not on a user request |
| Unsourced causal edge | Load fails |
| Intervention with no evidence | Load fails |
| Implausible extracted value (pH 47) | Dropped, valid siblings in the same message kept |
| Corpus edited after index build | Index detected as stale and rebuilt |
| chromadb / sentence-transformers missing | Warned, falls back to built-in backends |
| Site data almost empty | Asks ranked questions; asserts no recommendation |

---

## 7. Known limitations

Stated plainly, because a system that reports on scientific uncertainty should be
honest about its own.

- **Effect-size ranges are literature-derived approximations**, not site-specific
  predictions. They are the right order of magnitude for the conditions the cited
  studies cover; they are not a substitute for local trials.
- **The causal graph is hand-built.** Edge strengths are expert-assigned
  elasticity-like weights informed by the cited work, not fitted coefficients.
  They encode direction, relative importance and lag defensibly; they are not
  calibrated for quantitative prediction.
- **Secondary effect magnitudes are propagation strengths, not measured
  quantities.** They are labelled as such everywhere they appear.
- **Geo-coordinates are context, not a data source.** There is no live soil-grid
  or remote-sensing lookup; coordinates are recorded and used for regional
  context only. Wiring SoilGrids or a rainfall raster in behind
  `metrics.infer_context()` is the obvious next step.
- **The corpus is global-generalist.** It is weighted toward dryland
  agro-ecosystems, temperate arable and tropical restoration, because that is
  where the strongest quantitative meta-analytic evidence sits. Boreal, alpine
  and freshwater systems are thin.
- **Sessions are in-memory.** Deliberate — they are short-lived conversational
  state — but horizontal scaling needs the `SessionStore` interface pointed at
  Redis.
- **`richness_baselines.csv` is indicative.** Real baselines are regional and
  survey-protocol-dependent; these bands are a screening aid.
