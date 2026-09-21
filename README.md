# Biodiversity Intelligence Engine

**Darukaa.Earth AI Biodiversity Intelligence Challenge**

An AI environmental scientist, not a chatbot. It diagnoses what is limiting
biodiversity on a specific piece of land by reasoning across soil, water, climate,
land-use structure and human pressure **together**, then prescribes interventions
with quantified effect sizes, explicit causal mechanisms, time horizons,
confidence levels and real citations.

```
73 evidence cards  ·  71 distinct citations, 64/64 DOIs machine-verified
45 causal edges    ·  every one sourced to specific evidence cards
41 interventions   ·  18 flagged non-obvious · none loadable without evidence
27 diagnostic rules + graph-traced limiting-factor analysis
116 tests, all passing, all offline, no API key required
```

---

## The central design decision

The brief rules out "generic LLM-only solutions". This system takes that
literally:

> **The language model cannot produce a recommendation, an effect size, a
> citation, a confidence level or a time horizon.**

All of those are computed by a deterministic engine over a curated knowledge
base. The model does exactly two bounded jobs — parsing free text into a typed
schema, and phrasing the engine's output — and both have a working non-LLM
fallback.

So `ANTHROPIC_API_KEY` is **optional**. With no key, the system still extracts
site data from prose, diagnoses, retrieves evidence, reasons over the causal
graph, cites sources and answers in full structured markdown. The deterministic
path is the reference implementation; the LLM is optional polish.

Three things follow, and they are the reason the architecture is shaped this way:

1. **It is verifiable.** Same profile in, same assessment out. The test suite
   asserts on scientific reasoning, not on a sampler's mood.
2. **It cannot hallucinate a citation.** Every reference is a row in a
   version-controlled corpus, and `scripts/verify_citations.py` resolves all of
   them against Crossref and doi.org.
3. **A reviewer can run it in 60 seconds** with no key and no network.

---

## Quick start

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# The brief's worked example, end to end, no API key:
PYTHONPATH=src python -m bioai.cli demo 1
```

Other entry points:

```bash
PYTHONPATH=src python -m bioai.cli chat                          # interactive session
PYTHONPATH=src python -m bioai.cli assess --json examples/semi_arid_wheat.json
PYTHONPATH=src python -m bioai.cli search "do trees reduce water in drylands?"
PYTHONPATH=src python -m bioai.cli stats                         # knowledge base composition
PYTHONPATH=src python -m bioai.cli demo 2                        # scenarios 1-5

PYTHONPATH=src uvicorn bioai.api.app:app --reload                # API + web client
#   http://localhost:8000        minimal web client
#   http://localhost:8000/docs   OpenAPI
```

To enable natural-language parsing and narration, `cp .env.example .env` and set
`ANTHROPIC_API_KEY`. Nothing else changes; `/healthz` reports which mode is
active.

---

## Worked example — the brief's own case

**Input** (free text, turn 2 of a conversation):

> "Soil organic carbon is 0.3%, rainfall is low at about 420 mm, I grow
> monoculture wheat, semi-arid region, pH 5.2. Nearest scrub is 1.5 km away, I
> spray 4 times a season and burn the stubble."

**The engine's top recommendation is not "use cover crops".** It is:

### Faidherbia albida parkland at low density · non-obvious

**What to do** — Retain and protect naturally regenerating *Faidherbia albida*,
or plant at 10–40 stems per hectare within the species' native range, and crop
directly beneath the canopy. Protect young stems from browsing for three seasons.

**Why it works** — *Faidherbia* drops its leaves as the rainy season begins, so it
does not shade or compete for water during crop growth, while the leaf fall
delivers a nitrogen-rich mulch exactly when the crop needs nitrogen. Its deep
taproot draws water from below the crop root zone. This phenological
complementarity is why the usual tree–crop competition trade-off does not apply —
and why species selection dominates generic canopy-cover targets.

| Metric | Direction | Expected change | Time to effect | Basis |
| --- | --- | --- | --- | --- |
| soil total nitrogen | increase | +20–60% relative [Garrity et al. 2010] | 4–12 yr | primary (cited study) |
| soil organic carbon | increase | +15–40% relative [Garrity et al. 2010] | 5–15 yr | primary (cited study) |
| erosion severity | decrease | −30–70% relative [Kuyah et al. 2019] | 2–6 yr | primary (cited study) |
| soil moisture | increase | +10–30% relative [Ilstedt et al. 2016] | 3–8 yr | primary (cited study) |
| earthworm abundance | increase | propagation strength 0.65 via 1 causal step | 1.5–3 yr | secondary (causal model) |
| bulk density (compaction) | decrease | propagation strength 0.45 via 1 causal step | 2–4 yr | secondary (causal model) |
| species richness | increase | propagation strength 0.40 via 1 causal step | 3–6 yr | secondary (causal model) |

**Causal pathway** — `soil organic carbon -> +earthworm abundance`
**Horizon** — long (5+ years) · **Confidence** — high (0.73) · **Cost** — low ·
**Reversibility** — moderate

*Confidence limited by:* site data only 33% complete; 3 of 7 impacts are
model-propagated secondary effects, not directly measured for this intervention.

**Risks** — slow to establish; not native everywhere, so introducing it outside
its range carries invasion and provenance risk; browsing pressure requires
protection during establishment.

Then, in order: **rainwater harvesting pits charged with manure** (structure and
amendment together, because either alone performs far worse), **stopping stubble
burning**, and **farmer-managed natural regeneration** of existing rootstock —
which exploits root systems already adapted to the site rather than paying for
nursery seedlings that mostly die.

Note what the engine **refuses** to recommend here. The full
conservation-agriculture package is ruled out because the site burns its
residues, so the package cannot be implemented as specified — and the brief's own
evidence is that no-till *without* residue retention carries a 5.7% yield
penalty. That competing use of the residue has to be resolved first. Silvopasture,
rotational grazing and native grass reseeding are ruled out on land use. The
response says so, with the reason.

It also reports that Faidherbia, the zai pits and FMNR are mutually
**reinforcing**, and elsewhere flags genuine **conflicts** (zai pits and cover
crops compete for the same soil water) and **sequencing** constraints (pH
correction before legume-based measures; drainage before gypsum).

Run it yourself: `PYTHONPATH=src python -m bioai.cli demo 1`

---

## How the multi-metric reasoning actually works

This is the differentiator the brief asks for, so here is the mechanism rather
than a claim.

### 1. A signed, weighted, lagged causal graph

`src/bioai/knowledge/datasets/causal_graph.yaml` holds 45 edges. Each carries a
sign, an elasticity-like strength, a lag in years, a written mechanism, and the
**evidence card ids that justify it**. The loader refuses to start if any edge is
unsourced.

The graph is used three ways:

- **forward** — an intervention moves metric M; what else moves, by how much, and
  after how long? This is how a cited soil-carbon effect becomes a stated
  consequence for earthworms and then for species richness, with the lag
  accumulated along the path.
- **backward** — metric T is degraded; which upstream metrics can account for it?
  Each candidate is scored by `edge_strength × how_degraded_it_actually_is`, so an
  influential but *healthy* upstream metric explains nothing, while one that is
  both influential and degraded is named the limiting factor (Liebig's law of the
  minimum).
- **narration** — the chain is emitted as the recommendation's stated pathway.

Three edge features carry most of the intelligence:

| Feature | What it prevents |
| --- | --- |
| `non_monotonic` + `optimum` | Groundwater recharge peaks at *intermediate* tree cover. Below the optimum trees help; above it they hurt. With no site value the engine reports `context_dependent` and asks, rather than guessing. |
| `saturating_at` | Promising a 0.3%-SOC gain to a peat soil. |
| `threshold` | Telling a nitrogen-*deficient* smallholder to cut nitrogen. The edge from nitrogen loading to richness loss is inert below ~80 kg N/ha/yr. |

### 2. Diagnosis strictly before prescription

The engine never jumps from a symptom to an intervention. It first produces
ranked `Diagnosis` objects, each naming a limiting factor and the *interaction*
that makes it limiting. An intervention becomes a candidate only if a diagnosis it
`addresses` actually fired.

### 3. Leverage, not just evidence strength

A measure with excellent literature support that improves an already-healthy
metric scores low, because priority is
`3.0·severity + 2.0·leverage + 1.6·breadth + 1.5·evidence + 0.7·novelty − cost − slowness`
where leverage is how degraded the metrics it moves currently are.

### 4. Primary and secondary effects are never blended

A quantified effect from a cited study and a model-propagated consequence are
different epistemic objects. Secondary effects carry their pathway and are
labelled `secondary (causal model)` everywhere they appear, and the confidence
block says how many of the impacts are modelled rather than measured.

### The test that proves it reasons rather than pattern-matches

`tests/test_reasoning.py::test_opposite_sites_get_opposite_nitrogen_advice`
feeds the engine two mirror-image sites — nitrogen-*deficient* semi-arid cropland
and nitrogen-*saturated* temperate cropland — and asserts they receive opposite
prescriptions:

| | Semi-arid depleted | Temperate intensive |
| --- | --- | --- |
| Diagnosis | `nitrogen_limited` | `nutrient_enrichment` |
| Prescription | *add* biological N (legumes, N-fixing trees) | *re-time and reduce* N, restore structure |
| `nitrogen_rate_optimization` | explicitly excluded | selected |

Both are "low biodiversity cropland with a nitrogen issue". A similarity-search
system gives them the same answer.

---

## Knowledge system

Three kinds of knowledge, separated because they are used differently. Full
contract in [`docs/KNOWLEDGE_SCHEMA.md`](docs/KNOWLEDGE_SCHEMA.md).

### Evidence cards — 73 cards, `knowledge/corpus/*.yaml`

A card is not a text chunk. It is a structured, quantified, conditional finding:

```yaml
- id: ev-cover-crop-soc
  claim: >
    Cover crops raise topsoil organic carbon by roughly 0.20-0.60 t C/ha/yr
    (about +8% to +25% relative SOC stock over 5-10 years), with legume mixes
    at the upper end because they add nitrogen alongside carbon.
  mechanism: >               # WHY it is true - this is what transfers to new sites
    Continuous living roots deliver labile carbon as rhizodeposits directly into
    the aggregate-forming zone, where it is occluded in microaggregates rather
    than respired. Legumes lift the N supply, which lowers the C:N ratio of the
    residue and raises microbial carbon-use efficiency, so a larger fraction of
    fixed carbon is stabilised as microbial necromass instead of lost as CO2.
  effects:                   # machine-readable, so the engine COMPOSES effects
    - metric: soil_organic_carbon_pct
      direction: increase
      relative_change_pct: [8, 25]
      time_to_effect_years: [2.0, 5.0]
  applies_when:              # the applicability envelope
    - field: annual_rainfall_mm
      min: 300
      note: "below ~300 mm the cover crop competes for water with the cash crop"
  caveats:                   # become the recommendation's stated risks
    - "Termination by tillage can re-mineralise much of the accumulated carbon."
  citation:
    authors: "Poeplau, C. & Don, A."
    year: 2015
    venue: "Agriculture, Ecosystems & Environment 200:33-41"
    doi: "10.1016/j.agee.2014.10.024"
    type: meta_analysis      # drives the confidence calculus
```

Corpus composition — 20 meta-analyses, 18 systematic reviews, 15 field studies,
10 global assessments (IPCC SRCCL, IPCC AR6 WGIII Ch.7, IPBES 2019, FAO SWSR,
UNCCD GLO), 6 long-term experiments, 3 models, 1 technical guideline. **66% is
meta-analysis, systematic review or global assessment**, which is what justifies
the confidence levels the system reports.

### Structured reference datasets — `knowledge/datasets/*.csv`

- **`metric_bands.csv`** (73 rows) — the interpretation layer. Turns `0.3` into
  `critical` *with a sourced reason*: "below the ~1% threshold at which aggregate
  stability, water-holding capacity and soil faunal habitat decline together
  rather than proportionally".
- **`richness_baselines.csv`** (22 rows) — expected richness by land use × climate
  zone × taxon, so an observed count is judged against a **land-use-matched**
  baseline rather than against pristine reference.
- **`zone_reference.csv`** — agro-ecological zones for context inference.

### Intervention catalogue — 41 interventions

With `targets`, `addresses`, `requires`, `contraindications`, cost,
reversibility, establishment period, a **monitoring plan**, declared
synergies/conflicts and a novelty flag. An intervention with no supporting
evidence card **cannot load**, so nothing is recommendable that cannot be cited.

### Retrieval pipeline

```
score = 0.45 · dense(query, card)          # hashed-TFIDF or MiniLM, min-max normalised
      + 0.30 · bm25(query, card)           # rare terms: sodicity, zai, Faidherbia
      + 0.25 · condition_match(site, card) # ← what generic RAG does not have
      × 0.15  if any applies_when predicate is VIOLATED
```

The third signal is the important one. *"Biodiversity is declining on my land"* is
roughly equally similar to a Sahel agroforestry card and a temperate
nitrogen-deposition card; only the site profile breaks the tie. Condition matching
is **three-valued** — satisfied / violated / unknown — and unknown gets half
credit, because absence of information about a site is not evidence against a
card.

Violated evidence is penalised and partitioned, not silently dropped, because
*"this popular measure does not apply to you, because your rainfall is below
350 mm"* is often the most decision-relevant output.

**Measured relevance** on a 12-query expert-labelled set: **recall@1 = 67%,
recall@3 = 92%**, held as a regression floor by
`tests/test_retrieval.py::test_relevance_baseline`.

Inspect it directly — the pipeline is auditable from outside, not just described:

```bash
PYTHONPATH=src python -m bioai.cli search "trees and water in drylands" --profile examples/semi_arid_wheat.json
curl -s localhost:8000/knowledge/search -H 'content-type: application/json' \
  -d '{"query":"grazing management","profile":{"land_use":"cropland"},"include_violated":true}'
curl -s localhost:8000/knowledge/graph     # the whole causal graph, with sources
```

**Backends are swappable.** The default is a dependency-free persistent numpy
store with exact cosine plus a deterministic hashed-TF-IDF embedder — at 73 cards
exact search is faster *and* more accurate than ANN, so the portable choice is
also the correct one. `BIOAI_VECTOR_BACKEND=chroma` and `BIOAI_EMBEDDER=minilm`
swap in ChromaDB and sentence-transformers; both fall back with a warning rather
than crashing if the optional dependency is absent
(`pip install -r requirements-optional.txt`).

---

## Conversational intelligence

### Clarifying questions chosen by counterfactual simulation

Most systems ask for whatever field is empty, in a fixed order. This one
estimates the **decision value** of each unknown field:

```
for each unknown field:
    for each plausible probe value (low band / high band, worst / best category):
        re-run diagnosis + candidate generation with that value imputed
        measure how far the recommendation set moved
gain = 0.55·mean_divergence + 0.25·spread + 0.20·structural_prior
```

The **spread** term is the real signal: if the low case and the high case lead to
the same advice, the answer decides nothing and the question is not worth a turn.
This runs entirely on the deterministic layer — no LLM, no retrieval — so it
costs ~50 ms and runs on every turn.

```
You > Biodiversity is declining on my land

1. What is the soil pH?                                   [gain 0.67]
   Why it changes the answer: pH gates whether legume-based nitrogen and carbon
   measures can work at all, so it changes the order of operations rather than
   just the choice.
2. What is the soil organic carbon in the topsoil?        [gain 0.62]
3. What is the average annual rainfall?                   [gain 0.62]
   Why: rainfall decides whether added tree cover raises or lowers water
   availability, which flips the sign of the agroforestry recommendation.
```

### Memory is the profile, not the transcript

`dialogue/state.py` keeps the cumulative `SiteProfile` as primary state, so:

- turn 7 reasons over a number given in turn 2 without re-reading anything;
- *"actually, the pH is 6.4"* **overwrites the slot** rather than leaving two
  contradictory claims in a context window;
- an inferred value never overwrites a user-stated one — `provenance` tracks
  `user` / `inferred` / `regional_default` and the renderer labels them
  (`dry season length: 5 months (regional default)`).

### The ask-or-advise policy is not a hard gate

Ask below 18% completeness, or when nothing is recommendable and the top question
has real information gain — but **never more than twice in a row**, so the system
cannot get stuck interrogating a user who does not have the numbers. "Just tell
me", "I don't know" and "assume" stop the questioning immediately. Even when
advising, outstanding questions are carried at the end.

### Deterministic natural-language extraction

Units, negation and qualitative phrasing, with no LLM:

| Input | Extracted |
| --- | --- |
| `"organic matter is 3.4%"` | `soil_organic_carbon_pct: 1.97` (÷1.724 Van Bemmelen) |
| `"nearest woodland is 1.5 km"` | `distance_to_natural_habitat_m: 1500` |
| `"no invasive weeds"` | `invasive_species_present: false` (negation beats keyword) |
| `"rainfall is low"` | `annual_rainfall_mm: 420`, provenance `inferred` |
| `"23.5 S, 46.6 W"` | `geo: {latitude: -23.5, longitude: -46.6}` |
| `"I rotate 3, 4 crops"` | **not** read as coordinates |
| `"the pH is 47 and rainfall is 650 mm"` | pH dropped as implausible, rainfall kept |

---

## Input and output

**Input** — free text, structured JSON (`POST /assess`, any subset of ~45
fields), or both in the same turn. Geo-coordinates are accepted in decimal and
N/S/E/W forms and recorded as regional context.

**Output** — every recommendation carries: what to do · why it works (mechanism) ·
causal pathway · impacted metrics with direction, magnitude, time-to-effect and
primary/secondary basis · time horizon · confidence with explicit drivers *and*
limiters · full citations with DOIs · preconditions · risks and trade-offs ·
synergies and conflicts · a monitoring plan. Plus set-level interactions,
sequencing constraints, and what was ruled out and why.

Confidence is **capped by data completeness** (`0.45 + 0.5·min(completeness/0.6, 1)`):
strong literature on a barely-described site does not justify a confident
recommendation.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | multi-turn conversation; text and/or structured profile |
| `POST` | `/assess` | one-shot structured assessment, skips the dialogue policy |
| `GET` | `/session/{id}` | inspect accumulated memory, transcript, fields asked vs answered |
| `DELETE` | `/session/{id}` | clear a session |
| `POST` | `/knowledge/search` | retrieval layer with per-signal scores and site-condition verdicts |
| `GET` | `/knowledge/stats` | corpus composition and the full citation list |
| `GET` | `/knowledge/graph` | the causal graph with mechanisms and sources |
| `GET` | `/knowledge/interventions` | catalogue with predicates and supporting cards |
| `GET` | `/healthz` | knowledge base size, retrieval backend, LLM status |
| `GET` | `/` | web client |

```bash
curl -s localhost:8000/assess -H 'content-type: application/json' -d '{
  "profile": {
    "soil_organic_carbon_pct": 0.3, "annual_rainfall_mm": 420,
    "land_use": "cropland", "cropping_system": "monoculture",
    "primary_crop": "wheat", "soil_ph": 5.2,
    "pesticide_applications_per_season": 4,
    "distance_to_natural_habitat_m": 1500,
    "geo": {"latitude": 26.9, "longitude": 75.8}
  },
  "explain_retrieval": true
}' | jq '.assessment.recommendations[0]'
```

`explain_retrieval: true` returns the fusion weights, per-signal scores, matched
and violated site predicates, and the full scoring trace for every
recommendation.

---

## Project layout

```
src/bioai/
  schemas.py                    typed contracts; nothing crosses a module boundary untyped
  config.py                     env-driven settings; every default works offline
  knowledge/
    corpus/                     soil · biodiversity · water_climate · land_use · human_impact
    datasets/                   metric_bands · causal_graph · interventions · baselines · zones
    loader.py                   strict validation + referential integrity
    embeddings.py               hashed-TFIDF (default) | MiniLM (optional)
    vectorstore.py              numpy (default) | ChromaDB (optional)
    lexical.py                  BM25 from scratch
    retriever.py                hybrid, site-conditioned, fully traced
  reasoning/
    metrics.py                  banding, composite indices, coverage, context inference
    causal.py                   forward propagation · upstream tracing · pathway narration
    diagnostics.py              27 rules + graph-traced limiting-factor analysis
    recommend.py                candidates → filter → evidence → effects → confidence → rank
    engine.py                   the deterministic pipeline
  dialogue/
    extract.py                  NL → schema (rules first, LLM for gaps only)
    clarify.py                  value-of-information question selection
    state.py                    structured session memory
    render.py                   complete markdown, no LLM required
    orchestrator.py             ask-or-advise policy
  llm/                          client with hard offline guarantee + bounded prompts
  api/app.py                    FastAPI
  web/index.html                single-file web client, no framework, no build
tests/                          116 tests: knowledge · retrieval · reasoning · dialogue · API
docs/                           ARCHITECTURE.md · KNOWLEDGE_SCHEMA.md
scripts/                        build_index.py · verify_citations.py
examples/                       6 site profiles covering contrasting biomes
```

---

## Tests

```bash
python -m pytest tests/ -q        # 116 passed in ~14s, no network, no API key
```

| File | What it protects |
| --- | --- |
| `test_knowledge.py` (15) | every claim quantified and cited; every causal edge sourced; every intervention citeable; DOIs well-formed; bands contiguous; corpus weighted toward strong evidence |
| `test_retrieval.py` (19) | measured relevance floor; site-conditioning changes ranking; violated evidence excluded from support but retrievable; index persistence and staleness; BM25 and vector-store correctness |
| `test_reasoning.py` (34) | multi-metric diagnoses; **opposite sites get opposite advice**; non-monotonic edges flip sign with site value; saturation attenuates propagation; contraindications exclude inapplicable measures; ≥3 metrics per recommendation; confidence capped by completeness; determinism |
| `test_dialogue.py` (33) | unit conversion, negation, implausible-value rejection; questions ranked by information gain and adapted to the site; memory accumulates and corrections overwrite; asks then advises then stops asking |
| `test_api.py` (15) | contracts, session lifecycle, validation, retrieval introspection |

Several of these caught real bugs during development — a falsy-zero TTL default,
saturation read off the wrong end of a causal edge, negation losing to a keyword
match, and an `include_violated` flag that silently did nothing.

### Citation verification (requires network, deliberately not in CI)

```bash
python scripts/verify_citations.py --check-urls
```

```
64/64 DOIs resolved with a matching title, 0 possible mismatch(es), 0 unresolvable.
7/7 report URLs live (FAO, IPCC, IPBES, UNCCD, UNEP, IFPRI).
```

Resolves against Crossref, falling back to doi.org content negotiation for
DataCite registrations. Not in CI because tests must pass offline and must not
depend on third-party uptime.

---

## Deployment / CI

**CI** (`.github/workflows/ci.yml`) — on every push and PR, across Python 3.10 /
3.11 / 3.12: validate the knowledge base, build the index, run the full suite,
lint with ruff, then build the Docker image and smoke-test `/healthz` and
`/assess` against the running container.

**Docker**

```bash
docker build -t bioai .
docker run -p 8000:8000 bioai            # no secrets needed
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... bioai
```

The index is built at image-build time, so the container starts warm and a
malformed evidence card fails the build rather than a user request.

**Hosting** — `render.yaml` is a one-click Render blueprint (`plan: free`,
health check `/healthz`, no secrets required). A `Procfile` covers
Railway/Heroku-style buildpacks. The image honours `$PORT`, so it also runs
unchanged on Fly.io and Cloud Run.

**Configuration** — see `.env.example`. Every setting has a working default and
the system needs no configuration at all to run.

---

## Known limitations

Stated plainly, because a system that reports on scientific uncertainty should be
honest about its own.

- **Effect-size ranges are literature-derived approximations**, not site-specific
  predictions. They are the right order of magnitude for the conditions the cited
  studies cover; they are not a substitute for local trials.
- **The causal graph is hand-built.** Edge strengths are expert-assigned weights
  informed by the cited work, not fitted coefficients. They encode direction,
  relative importance and lag defensibly; they are not calibrated for
  quantitative prediction.
- **Secondary effect magnitudes are propagation strengths**, not measured
  quantities — labelled as such wherever they appear.
- **Geo-coordinates are context, not a data source.** There is no live SoilGrids
  or remote-sensing lookup; coordinates are recorded and used for regional
  context only. Wiring a soil-grid or rainfall raster in behind
  `metrics.infer_context()` is the obvious next step.
- **The corpus is global-generalist**, weighted toward dryland agro-ecosystems,
  temperate arable and tropical restoration, because that is where the strongest
  quantitative meta-analytic evidence sits. Boreal, alpine and freshwater systems
  are thin.
- **Sessions are in-memory** — deliberate for short-lived conversational state,
  but horizontal scaling needs the `SessionStore` interface pointed at Redis.
- **`richness_baselines.csv` is a screening aid.** Real baselines are regional
  and survey-protocol-dependent.

---

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full pipeline, scoring
  formulas, failure behaviour
- [`docs/KNOWLEDGE_SCHEMA.md`](docs/KNOWLEDGE_SCHEMA.md) — the data contract, so
  a domain scientist can extend the corpus without touching Python
