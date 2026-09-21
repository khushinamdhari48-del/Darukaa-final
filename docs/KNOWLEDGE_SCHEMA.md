# Knowledge base schema

Everything in `src/bioai/knowledge/` is data, not code. This document is the
contract, so the corpus can be extended by a domain scientist without touching
Python.

All of it is validated at load time by `knowledge/loader.py`. **Validation is
strict and fails loudly**, because a silently mis-parsed evidence card would
produce a confidently wrong recommendation. Adding a card with a dangling
intervention reference, or an intervention with no supporting evidence, prevents
the application from starting.

---

## 1. Evidence cards — `corpus/*.yaml`

Each file is a YAML list of cards. Files are split by domain for human
convenience only; the loader concatenates them and enforces globally unique ids.

```yaml
- id: ev-cover-crop-soc                 # required, globally unique, kebab-case
  claim: >                              # required, >40 chars, MUST carry numbers
    Cover crops raise topsoil organic carbon by roughly 0.20-0.60 t C/ha/yr
    (about +8% to +25% relative SOC stock over 5-10 years), with legume mixes
    at the upper end because they add nitrogen alongside carbon.
  mechanism: >                          # required, >80 chars: WHY, not a restatement
    Continuous living roots deliver labile carbon as rhizodeposits directly into
    the aggregate-forming zone, where it is occluded in microaggregates rather
    than respired. Legumes lift the N supply, which lowers the C:N ratio of the
    residue and raises microbial carbon-use efficiency, so a larger fraction of
    fixed carbon is stabilised as microbial necromass instead of being lost as CO2.
  domain: soil                          # soil | biodiversity | climate | water |
                                        # land_use | human_impact | cross_cutting
  effects:                              # machine-readable, composable
    - metric: soil_organic_carbon_pct   # must be a SiteProfile field name
      direction: increase               # increase | decrease
      absolute_change: "+0.20 to +0.60 t C/ha/yr in 0-30 cm"
      relative_change_pct: [8, 25]      # [low, high], low <= high
      time_to_effect_years: [2.0, 5.0]  # [earliest, typical], 0 allowed
      measurement_context: "0-30 cm, global meta-analysis of 139 comparisons"
  applies_when:                         # applicability envelope (see 1.1)
    - field: land_use
      allowed: [cropland, orchard, plantation, agroforestry]
    - field: annual_rainfall_mm
      min: 300
      note: "below ~300 mm the cover crop competes for water with the cash crop"
  caveats:                              # surfaced as the recommendation's risks
    - "Gains saturate as the soil approaches its carbon-stabilisation capacity."
    - "Termination by tillage can re-mineralise much of the accumulated carbon."
  citation:                             # required; DOI or URL mandatory
    authors: "Poeplau, C. & Don, A."    # "Last, F. M., Last, F. M. & Last, F."
    year: 2015                          # the ISSUE year, not the online-first year
    title: "Carbon sequestration in agricultural soils via cultivation of cover crops - A meta-analysis"
    venue: "Agriculture, Ecosystems & Environment 200:33-41"
    doi: "10.1016/j.agee.2014.10.024"
    type: meta_analysis                 # drives the confidence calculus (see 1.2)
  interventions: [legume_cover_crops, improved_fallow_legume]   # must exist
  tags: [cover crops, soil organic carbon, legume, rhizodeposition]
  text: >                               # required: the chunk that gets indexed
    A global meta-analysis of cover-crop experiments found a mean soil organic
    carbon accumulation rate of 0.32 t C/ha/yr in the top 30 cm [...]
```

### 1.1 `applies_when` — three-valued predicates

This is the field that makes retrieval site-aware rather than merely
similarity-based.

| Form | Meaning |
| --- | --- |
| `min: 300` | numeric field must be ≥ 300 |
| `max: 1200` | numeric field must be ≤ 1200 |
| `min: 400`<br>`max: 1200` | inclusive range |
| `allowed: [a, b]` | categorical must be one of these |
| `forbidden: [deep]` | categorical must not be one of these |
| `note: "..."` | shown to the user when the predicate is *violated* |

Each predicate evaluates to **satisfied**, **violated**, or **unknown** (the site
has no value for that field). Unknown receives half credit: absence of
information about a site is not evidence against a card.

A card with **any violated** predicate has its score multiplied by 0.15 and is
excluded from recommendation support. It remains retrievable via
`include_violated: true`, because "this does not apply to you, because X" is
useful output.

An **empty** `applies_when` means universally applicable, and scores a neutral
0.5 rather than 1.0 — so a card demonstrably matched to the site outranks a
generic one.

### 1.2 `citation.type` — inferential strength

Sets the ceiling on the confidence of any recommendation resting on this card.

| Type | Weight |
| --- | --- |
| `meta_analysis` | 1.00 |
| `systematic_review` | 0.95 |
| `global_assessment` | 0.90 |
| `long_term_experiment` | 0.85 |
| `model` | 0.70 |
| `field_study` | 0.65 |
| `technical_guideline` | 0.60 |
| `dataset` | 0.55 |

`tests/test_knowledge.py::test_corpus_is_weighted_toward_strong_evidence`
requires >50% of cards to be meta-analysis, systematic review or global
assessment. Current corpus: **73 cards, 71 distinct citations, 66% strong**.

### 1.3 Citation integrity

Every citation must carry a DOI or, for institutional reports that have none, a
URL. `scripts/verify_citations.py` resolves all of them over the network:

```bash
python scripts/verify_citations.py --check-urls
# 64/64 DOIs resolved with a matching title, 0 mismatches, 0 unresolvable
# 7/7 report URLs live
```

It tries Crossref first, then doi.org content negotiation (which covers DataCite
registrations such as Zenodo). A ±1-year difference between the recorded year and
the registered year is tolerated as an online-first vs issue-date artefact.

This script is deliberately **not** in CI: tests must pass offline and must not
depend on third-party uptime.

---

## 2. Metric bands — `datasets/metric_bands.csv`

The interpretation layer. Turns a number into a sourced verdict.

```csv
metric,band,low,high,verdict,interpretation,source
soil_organic_carbon_pct,very_low,0,0.6,critical,"Severely depleted; aggregate stability, water retention and soil faunal habitat are all compromised simultaneously.",FAO 2017 Soil Organic Carbon: the hidden potential
```

| Column | Rule |
| --- | --- |
| `metric` | a `SiteProfile` field name |
| `low`, `high` | half-open `[low, high)`; **bands must be contiguous** per metric |
| `verdict` | `critical` \| `poor` \| `moderate` \| `good` \| `high` |
| `interpretation` | why this band matters — shown to the user |
| `source` | required; short citation form |

Verdicts map to condition scores non-linearly — `critical → 0.05`, `poor → 0.30`,
`moderate → 0.55`, `good → 0.82`, `high → 0.90` — so a single critical metric
dominates the composite index rather than being averaged away.

**Integer-count metrics** (pesticide applications) express bands as half-open
real intervals around the integer boundaries (`0–0.5`, `0.5–2.5`, `2.5–6.5`) so
that contiguity holds and no band is degenerate.

Contiguity is asserted by
`tests/test_knowledge.py::test_metric_bands_are_contiguous_and_ordered`.

---

## 3. Causal graph — `datasets/causal_graph.yaml`

```yaml
edges:
  - source: soil_organic_carbon_pct     # SiteProfile field
    target: soil_moisture_pct           # SiteProfile field
    sign: positive                      # positive | negative | non_monotonic
    strength: 0.55                      # (0, 1] elasticity-like weight
    lag_years: 1.0                      # >= 0, accumulates along a path
    mechanism: >                        # required, >60 chars
      Organic matter holds several times its own mass in water and stabilises
      the aggregates that maintain the pore-size distribution responsible for
      plant-available water.
    evidence: [ev-fao-soc-hidden-potential, ev-lal-soil-carbon-cobenefits]
    # optional modifiers, all evaluated against the SOURCE metric's value:
    optimum: 15.0        # required when sign: non_monotonic
    saturating_at: 25.0  # above this, strength x 0.35
    threshold: 80.0      # below this, strength x 0.25 (edge is inert)
```

**`evidence` is mandatory and validated.** An unsourced edge, or one citing a
card that does not exist, fails the load. This is the mechanism that keeps the
causal model from drifting into unfounded assertion.

### Semantics of the three modifiers

All three are properties of where the **source** metric currently sits on its
dose–response curve.

- **`sign: non_monotonic` + `optimum`** — the sign depends on which side of the
  optimum the site is on. Below: positive. Above: negative. **No site value:
  `context_dependent`**, which is reported honestly and becomes a clarifying
  question rather than a guess.
- **`saturating_at`** — diminishing returns above this source level. Adding
  habitat to an already habitat-rich landscape, or carbon to a peat soil.
- **`threshold`** — the edge barely operates below this source level. Nitrogen
  only suppresses plant richness once it has ceased to be limiting; below the
  threshold, adding nitrogen is beneficial.

### How the graph is used

| Operation | Purpose |
| --- | --- |
| `forward_effects()` | secondary consequences of an intervention, with accumulated lag |
| `upstream_causes()` | backward trace to find the limiting factor behind a degraded outcome |
| `pathway_text()` | renders a path as `soil moisture -> +soil organic carbon -> +earthworm abundance` |

Cycles are expected (soil carbon ↔ moisture ↔ richness). Each metric is kept once
at its strongest path; depth is capped at 3; effects below strength 0.12 are
pruned.

---

## 4. Interventions — `datasets/interventions.yaml`

```yaml
interventions:
  - id: legume_cover_crops
    title: Legume-based multi-species cover crop in the fallow window
    what_to_do: >                       # specific and actionable - no "use
                                        # sustainable practices"
      Sow a legume-dominant mixture into the fallow window, terminate by rolling
      or grazing rather than by tillage, and leave the residue on the surface.
      Target 3-4 functional groups rather than a single species.
    targets: [soil_organic_carbon_pct, soil_nitrogen_pct, earthworm_count_per_m2]
    addresses: [soc_depleted, nitrogen_limited, soil_biology_collapsed]
    requires:                           # unknown => precondition, NOT a block
      - field: land_use
        allowed: [cropland, orchard, plantation, agroforestry]
    contraindications:                  # true => hard exclusion, with a reason
      - field: annual_rainfall_mm
        max: 250
        note: "below ~250 mm the cover crop depletes the water the cash crop needs"
    cost_intensity: low                 # low | moderate | high
    reversibility: easy                 # easy | moderate | hard
    time_horizon: medium                # short | medium | long (lower bound)
    establishment_years: [0.5, 1.0]     # floors the reported time-to-effect
    monitoring:                         # required: how to verify it worked
      - "Soil organic carbon at 0-30 cm with bulk-density correction, 3-yearly"
      - "Earthworm counts by hand sorting of 20x20x20 cm blocks, 5 replicates"
    synergies: [reduced_tillage, mycorrhiza_management]
    conflicts: [rainwater_harvesting_zai]   # must be declared on BOTH sides
    novelty: standard                   # standard | non_obvious
```

### `requires` vs `contraindications`

The distinction is the difference between "does not apply" and "we do not yet
know whether it applies".

| Predicate | Evaluates | Result |
| --- | --- | --- |
| `requires` | **false** | candidate dropped, reason reported to the user |
| `requires` | **unknown** | kept, becomes a stated precondition, raises the value of asking about that field |
| `contraindications` | **true** | candidate dropped, reason reported |

### `addresses` — the diagnosis vocabulary

An intervention only becomes a candidate if one of the diagnoses it `addresses`
actually fired, or if it directly targets a metric the causal graph identified as
limiting. Valid codes are those emitted by `reasoning/diagnostics.py`:

```
soc_depleted            acidity_constraint      alkalinity_constraint
salinity_constraint     compaction              soil_biology_collapsed
erosion_risk            bare_soil_exposure      crusted_surface
water_partitioning_loss water_scarcity_for_fauna  waterlogging
monoculture_risk        structural_simplification  habitat_isolation
fragmentation           degraded_land           invasive_dominance
pollinator_deficit      natural_enemy_deficit   pesticide_dependency
nutrient_enrichment     nitrogen_limited        overgrazing
pollution_pathway       deforestation_active    richness_below_baseline
```

Plus synthetic `limiting::<target>::<cause>` codes generated by the graph trace.

### `novelty: non_obvious`

Marks interventions a non-specialist would be unlikely to propose, or where the
*correct form* of the action is counter-intuitive. Worth +0.7 in priority
scoring, because the brief explicitly rejects obvious advice. Current corpus:
**18 of 41 interventions** are flagged non-obvious — for example:

- Faidherbia parkland (reverse phenology means no tree–crop competition)
- assisted natural regeneration *in preference to* planting
- pollinator *nesting* substrate, not just flowers
- managing resident mycorrhizal networks rather than buying inoculants
- converting the lowest-yielding 3–8% of a field, not a uniform strip
- drainage **before** gypsum on sodic soils

---

## 5. Richness baselines — `datasets/richness_baselines.csv`

```csv
land_use,climate_zone,taxon,expected_min,expected_max,unit,notes,source
cropland,semi_arid,vascular_plants,8,25,species per 100 m2,"Monoculture with herbicide use sits at the low end.",Newbold et al. 2015
```

Lets an observed species count be benchmarked against a **land-use-matched**
expectation rather than against pristine reference — the only defensible
comparison for a working farm. Falls back to the same land use in any zone when
no exact zone match exists, and the caller treats that as a weaker comparison.

These bands are a **screening aid**. Real baselines are regional and
survey-protocol-dependent.

---

## 6. Agro-ecological zones — `datasets/zone_reference.csv`

Maps rainfall to climate zone, used by `metrics.infer_context()` to fill
`climate_zone` and `dry_season_months` when the user has not supplied them.
Inferred values are tagged `inferred` or `regional_default` in
`SiteProfile.provenance`, rendered with an explicit label, and **never overwrite
a user-stated value**.

`montane` is temperature-defined, not rainfall-defined, and is therefore never
inferred from rainfall alone.

---

## 7. Adding to the knowledge base

```bash
# 1. Add the card / edge / intervention to the relevant YAML or CSV.
# 2. Validation + rebuild (fails loudly on any integrity problem):
python scripts/build_index.py --force

# 3. Integrity and reasoning tests:
python -m pytest tests/ -q

# 4. If you added a citation, verify it actually exists:
python scripts/verify_citations.py --check-urls
```

Checklist for a new evidence card:

- [ ] `claim` carries an effect size, not a direction only
- [ ] `mechanism` explains *why*, and would transfer to a site the study did not cover
- [ ] `effects[].metric` values are real `SiteProfile` field names
- [ ] `applies_when` states the envelope — where does this stop being true?
- [ ] `caveats` state what breaks it
- [ ] `citation` has a resolvable DOI or URL, and the **issue** year
- [ ] `citation.type` honestly reflects the study design
- [ ] `interventions` reference existing intervention ids
- [ ] `tags` include the rare technical terms a user might actually type
- [ ] if it justifies a new causal edge, add the edge and cite this card in it
