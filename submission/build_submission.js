// Builds the Darukaa.Earth hackathon submission document.
//   node submission/build_submission.js
const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  ExternalHyperlink,
  HeadingLevel,
  LevelFormat,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const ACCENT = "1F5C3F";
const INK = "1C1F1A";
const MUTED = "5C6158";
const RULE = "C9CCC3";
const CODE_BG = "F1F3EE";
const HEAD_BG = "E7EFE9";

// US Letter, in DXA (1440 = 1 inch)
const PAGE = { width: 12240, height: 15840 };
const MARGIN = 1080; // 0.75"
const CONTENT_WIDTH = PAGE.width - MARGIN * 2; // 10080

// Fill these in before submitting. They are rendered in a highlighted
// "action required" block so the document cannot be sent with placeholders
// left in unnoticed.
const REPO_URL = process.env.REPO_URL || "";
const DEMO_URL = process.env.DEMO_URL || "";

// ---------------------------------------------------------------------------
// building blocks
// ---------------------------------------------------------------------------

const t = (text, opts = {}) =>
  new TextRun({ text, font: "Calibri", size: 21, color: INK, ...opts });

const mono = (text, opts = {}) =>
  new TextRun({ text, font: "Consolas", size: 18, color: INK, ...opts });

const body = (children, opts = {}) =>
  new Paragraph({
    children: Array.isArray(children) ? children : [t(children)],
    spacing: { after: 140, line: 288 },
    ...opts,
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 380, after: 170 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 6 } },
    children: [new TextRun({ text, font: "Calibri", size: 27, bold: true, color: ACCENT })],
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 110 },
    children: [new TextRun({ text, font: "Calibri", size: 23, bold: true, color: INK })],
  });

const bullet = (children, level = 0) =>
  new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 90, line: 276 },
    children: Array.isArray(children) ? children : [t(children)],
  });

const codeBlock = (lines) =>
  lines.map((line, i) =>
    new Paragraph({
      children: [mono(line || " ")],
      shading: { type: ShadingType.CLEAR, fill: CODE_BG, color: "auto" },
      spacing: { after: i === lines.length - 1 ? 150 : 0, before: i === 0 ? 40 : 0, line: 240 },
      indent: { left: 200, right: 200 },
    })
  );

const link = (text, url) =>
  new ExternalHyperlink({
    link: url,
    children: [
      new TextRun({
        text,
        font: "Calibri",
        size: 21,
        color: "1155CC",
        underline: {},
      }),
    ],
  });

const monoLink = (text, url) =>
  new ExternalHyperlink({
    link: url,
    children: [
      new TextRun({ text, font: "Consolas", size: 18, color: "1155CC", underline: {} }),
    ],
  });

/** Table with dual DXA widths, as required for Google Docs compatibility. */
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== CONTENT_WIDTH) {
    throw new Error(`column widths sum to ${total}, expected ${CONTENT_WIDTH}`);
  }
  const cell = (content, width, opts = {}) =>
    new TableCell({
      width: { size: width, type: WidthType.DXA },
      margins: { top: 90, bottom: 90, left: 120, right: 120 },
      shading: opts.header
        ? { type: ShadingType.CLEAR, fill: HEAD_BG, color: "auto" }
        : undefined,
      children: [
        new Paragraph({
          spacing: { after: 0, line: 264 },
          children: Array.isArray(content) ? content : [t(content, { bold: !!opts.header })],
        }),
      ],
    });

  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      right: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideVertical: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, widths[i], { header: true })),
      }),
      ...rows.map(
        (r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i])) })
      ),
    ],
  });
}

/** A URL cell that is obviously unfilled when it is unfilled. */
const urlCell = (url) =>
  url
    ? [monoLink(url, url)]
    : [
        new TextRun({
          text: "▶ TO BE FILLED IN BEFORE SUBMITTING",
          font: "Calibri",
          size: 21,
          bold: true,
          color: "B3261E",
          highlight: "yellow",
        }),
      ];

const spacer = (after = 200) => new Paragraph({ children: [t("")], spacing: { after } });

// ---------------------------------------------------------------------------
// document
// ---------------------------------------------------------------------------

const children = [];

// ---- title block ----
children.push(
  new Paragraph({
    spacing: { after: 60 },
    children: [
      new TextRun({
        text: "Darukaa.Earth — AI Biodiversity Intelligence Challenge",
        font: "Calibri",
        size: 20,
        color: MUTED,
        allCaps: true,
      }),
    ],
  }),
  new Paragraph({
    spacing: { after: 100 },
    children: [
      new TextRun({
        text: "Biodiversity Intelligence Engine",
        font: "Calibri",
        size: 40,
        bold: true,
        color: ACCENT,
      }),
    ],
  }),
  new Paragraph({
    spacing: { after: 240 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 8 } },
    children: [
      new TextRun({
        text: "An evidence-grounded environmental advisory system: it diagnoses what is limiting biodiversity on a specific site by reasoning across soil, water, climate, land use and human pressure together, then prescribes interventions with quantified effect sizes, causal mechanisms, time horizons, confidence levels and verified citations.",
        font: "Calibri",
        size: 21,
        color: INK,
        italics: true,
      }),
    ],
  })
);

if (!REPO_URL || !DEMO_URL) {
  children.push(
    new Paragraph({
      spacing: { after: 220 },
      shading: { type: ShadingType.CLEAR, fill: "FFF4CE", color: "auto" },
      indent: { left: 160, right: 160 },
      border: {
        left: { style: BorderStyle.SINGLE, size: 18, color: "E8A800", space: 8 },
      },
      children: [
        new TextRun({
          text: "ACTION REQUIRED BEFORE SUBMITTING  ",
          font: "Calibri",
          size: 20,
          bold: true,
          color: "7A4E00",
        }),
        new TextRun({
          text:
            "Insert the GitHub repository URL and, if deployed, the live demo URL in the highlighted cells below and in section 1. Regenerate with:  REPO_URL=... DEMO_URL=... node submission/build_submission.js",
          font: "Calibri",
          size: 20,
          color: "7A4E00",
        }),
      ],
    })
  );
}

children.push(
  table(
    ["", ""],
    [
      ["Submitted by", "Khushi — khushi@hyperface.co"],
      ["Submission date", "18 September 2026"],
      ["Repository", urlCell(REPO_URL)],
      ["Live demo", urlCell(DEMO_URL)],
      ["Credentials required", "None. No API key, no database, no login."],
    ],
    [2600, 7480]
  ),
  spacer(260)
);

// ---- 1. submission links ----
children.push(h1("1. Submission links"));

children.push(h2("GitHub repository"));
children.push(body(urlCell(REPO_URL)));
children.push(
  body(
    "The repository is public, so no access invitations are required. If it is switched to private, reviewer access will be granted to ankita.dasgupta@darukaa.com, harsh.kumar@darukaa.com, utkarsh.gauniyal@darukaa.com and guneet.mutreja@darukaa.com."
  )
);

children.push(h2("Live demo"));
children.push(body(urlCell(DEMO_URL)));
children.push(
  table(
    ["Path", "What it shows"],
    [
      ["/", "Web client: conversation, structured JSON input, and a knowledge-retrieval inspector"],
      ["/docs", "Interactive OpenAPI documentation for every endpoint"],
      ["/healthz", "Knowledge base size, retrieval backend and LLM status"],
      ["/knowledge/stats", "Corpus composition and the full list of 71 citations"],
      ["/knowledge/graph", "The complete causal graph, with mechanisms and source cards"],
    ],
    [2300, 7780]
  ),
  spacer(160)
);
children.push(
  body([
    t("Note on the hosted demo: the Render free tier sleeps when idle, so the "),
    t("first request may take 30–50 seconds", { bold: true }),
    t(
      " to wake the container. Subsequent requests are fast. The deployment requires no secrets, so it can be recreated from the included "
    ),
    mono("render.yaml"),
    t(
      " blueprint at any time. If the demo is unavailable for any reason, the system runs locally in under a minute with no API key and no configuration — see section 5."
    ),
  ])
);

// ---- 2. what it does ----
children.push(h1("2. What the system does"));

children.push(
  body([
    t("Worked example — the brief's own case. ", { bold: true }),
    t("Given the free-text input:"),
  ])
);
children.push(
  ...codeBlock([
    '"Soil organic carbon is 0.3%, rainfall is low at about 420 mm, I grow',
    ' monoculture wheat, semi-arid region, pH 5.2. Nearest scrub is 1.5 km',
    ' away, I spray 4 times a season and burn the stubble."',
  ])
);
children.push(
  body([
    t("the engine's top recommendation is not "),
    t("“use cover crops”", { italics: true }),
    t(". It is a "),
    t("Faidherbia albida parkland at 10–40 stems per hectare", { bold: true }),
    t(
      " — because Faidherbia is reverse-phenological: it sheds its leaves as the rainy season begins, so it does not compete with the crop for water during growth, while the leaf fall delivers nitrogen exactly when the crop needs it. That phenological complementarity is why the usual tree–crop water trade-off does not apply at 420 mm, and it is the kind of species-specific reasoning a similarity-search system cannot reach."
    ),
  ])
);
children.push(
  body([
    t("Equally important is what the engine "),
    t("refuses", { bold: true, italics: true }),
    t(
      " to recommend. The full conservation-agriculture package is ruled out here because the site burns its residues, so the package cannot be implemented as specified — and no-till without residue retention carries a 5.7% yield penalty (Pittelkow et al. 2015, "
    ),
    t("Nature", { italics: true }),
    t(
      " 517:365-368). The response says so, with the reason. Silvopasture, rotational grazing and native grass reseeding are excluded on land use."
    ),
  ])
);

children.push(h2("Every recommendation carries"));
children.push(
  table(
    ["Field", "Example from the output above"],
    [
      ["What to do", "Specific and actionable — stems per hectare, pruning regime, browsing protection"],
      ["Why it works", "The mechanism, not a restatement: reverse phenology, nitrogen-rich litter timing, taproot depth"],
      ["Causal pathway", [mono("soil organic carbon -> +earthworm abundance")]],
      ["Impacted metrics", "7 metrics with direction, magnitude, time-to-effect, and whether the basis is a cited study or the causal model"],
      ["Time horizon", "Long (5+ years), floored by the establishment period so a slow measure is never sold as quick"],
      ["Confidence", "High (0.73), with explicit drivers AND limiters — here: site data only 33% complete; 3 of 7 impacts are model-propagated"],
      ["Evidence", "4 citations with DOIs, plus which site conditions confirmed applicability"],
      ["Risks / trade-offs", "Slow establishment; provenance and invasion risk outside native range; browsing pressure"],
      ["Monitoring plan", "Under-canopy vs open-field yield in the same field — the direct test of the claimed mechanism"],
      ["Set-level output", "Reinforcing pairs, genuine conflicts, and sequencing constraints across the whole recommendation set"],
    ],
    [2300, 7780]
  ),
  spacer(200)
);

children.push(h2("The test that shows it reasons rather than pattern-matches"));
children.push(
  body(
    "Two mirror-image sites are fed to the engine. Both are “low-biodiversity cropland with a nitrogen issue”, and a similarity-search system gives them the same answer:"
  )
);
children.push(
  table(
    ["", "Semi-arid, depleted", "Temperate, intensive"],
    [
      ["Soil organic carbon", "0.3%", "2.4%"],
      ["Nitrogen applied", "25 kg N/ha/yr", "220 kg N/ha/yr"],
      ["Diagnosis", [mono("nitrogen_limited")], [mono("nutrient_enrichment")]],
      ["Prescription", "Add biological nitrogen (legumes, N-fixing trees)", "Re-time and reduce nitrogen; restore structure"],
      ["Rate optimisation", "Explicitly excluded", "Selected"],
    ],
    [2500, 3790, 3790]
  ),
  spacer(160)
);
children.push(
  body([
    t("Asserted by "),
    mono("tests/test_reasoning.py::test_opposite_sites_get_opposite_nitrogen_advice"),
    t("."),
  ])
);

// ---- 3. architecture ----
children.push(h1("3. Architecture"));

children.push(h2("The central design decision"));
children.push(
  body([
    t("The brief rules out generic LLM-only solutions, so this system takes that literally: "),
    t(
      "the language model cannot produce a recommendation, an effect size, a citation, a confidence level or a time horizon",
      { bold: true }
    ),
    t(
      ". All of those are computed by a deterministic engine over a curated knowledge base. The model does exactly two bounded jobs — parsing free text into a typed schema, and phrasing the engine's output — and both have a working non-LLM fallback."
    ),
  ])
);
children.push(
  body([
    t("Consequently "),
    mono("ANTHROPIC_API_KEY"),
    t(" is "),
    t("optional", { bold: true }),
    t(
      ". With no key the system still extracts site data from prose, diagnoses, retrieves evidence, reasons over the causal graph, cites sources and answers in full structured markdown. Three properties follow, and they are the reason the architecture is shaped this way: it is "
    ),
    t("verifiable", { bold: true }),
    t(" (same input, same output, so tests assert on scientific reasoning); it "),
    t("cannot hallucinate a citation", { bold: true }),
    t(
      " (every reference is a row in a version-controlled corpus); and a reviewer can run it offline in under a minute."
    ),
  ])
);

children.push(h2("Pipeline"));
children.push(
  ...codeBlock([
    "free text ──┐",
    "            ├─► extract.py      regex + units + negation; LLM fills gaps only",
    "  JSON ─────┘                   (a rule-extracted number is never overridden)",
    "                 │",
    "                 ▼",
    "            state.py            MEMORY IS THE STRUCTURED PROFILE, NOT THE",
    "                                TRANSCRIPT, so a later correction overwrites",
    "                                the slot instead of contradicting it",
    "                 │",
    "                 ▼",
    "  ┌──────────────────────────────────────────────────────────────┐",
    "  │  reasoning/engine.py   deterministic · no network · no LLM   │",
    "  │   1. metrics.py      values ─► banded scores ─► 5 composite  │",
    "  │                      indices, each with its own coverage     │",
    "  │   2. diagnostics.py  27 rules + limiting-factor analysis     │",
    "  │                      traced BACKWARD through the causal graph│",
    "  │   3. recommend.py    candidates ─► contraindication filter   │",
    "  │                      ─► evidence ─► effect composition       │",
    "  │                      ─► confidence ─► priority ─► set-level  │",
    "  │   4. clarify.py      counterfactual simulation of what to ask│",
    "  └───────────┬──────────────────────────────┬───────────────────┘",
    "              ▼                              ▼",
    "   retriever.py  hybrid,          causal_graph.yaml  45 signed,",
    "   site-conditioned:              weighted, lagged edges — every",
    "     dense 0.45                   one sourced to evidence cards",
    "     BM25  0.30",
    "     site-condition 0.25",
    "              │",
    "              ▼",
    "   73 evidence cards / 71 citations · 41 interventions",
    "   73 metric bands · 22 richness baselines · 6 agro-ecological zones",
    "              │",
    "              ▼",
    "   render.py   complete markdown, always. The LLM (optional)",
    "               paraphrases it; engine output is kept verbatim below.",
  ])
);

children.push(h2("Multi-metric reasoning: the mechanism"));
children.push(
  body(
    "This is the brief's core differentiator, so it is implemented as an explicit model rather than as prompt instructions."
  )
);
children.push(
  bullet([
    t("A signed, weighted, lagged causal graph. ", { bold: true }),
    t(
      "45 edges, each with a sign, an elasticity-like strength, a lag in years, a written mechanism, and the evidence card IDs that justify it. The loader "
    ),
    t("refuses to start", { bold: true }),
    t(" if any edge is unsourced or cites a card that does not exist."),
  ])
);
children.push(
  bullet([
    t("Used forward", { bold: true }),
    t(
      " to derive secondary consequences with accumulated lag (a cited soil-carbon effect becomes a stated consequence for earthworms, then for species richness); "
    ),
    t("backward", { bold: true }),
    t(
      " to find the limiting factor, scoring each upstream metric by edge strength × how degraded it actually is — so an influential but healthy metric explains nothing (Liebig's law of the minimum)."
    ),
  ])
);
children.push(
  bullet([
    t("Three edge features carry most of the intelligence. ", { bold: true }),
    mono("non_monotonic"),
    t(" with an "),
    mono("optimum"),
    t(
      ": groundwater recharge peaks at intermediate tree cover, so below the optimum trees help and above it they hurt — and with no site value the engine reports "
    ),
    mono("context_dependent"),
    t(" and asks, rather than guessing. "),
    mono("saturating_at"),
    t(" stops it promising a 0.3%-carbon gain to a peat soil. "),
    mono("threshold"),
    t(" stops it telling a nitrogen-deficient smallholder to cut nitrogen."),
  ])
);
children.push(
  bullet([
    t("Diagnosis strictly before prescription. ", { bold: true }),
    t(
      "An intervention becomes a candidate only if a diagnosis it addresses actually fired, or if it targets a metric the graph identified as limiting."
    ),
  ])
);
children.push(
  bullet([
    t("Leverage, not just evidence strength. ", { bold: true }),
    t(
      "Priority = 3.0·severity + 2.0·leverage + 1.6·breadth + 1.5·evidence + 0.7·novelty − cost − slowness, where leverage is how degraded the metrics the measure moves currently are. This is what stops the engine recommending cover crops to a 4%-carbon soil."
    ),
  ])
);
children.push(
  bullet([
    t("Primary and secondary effects are never blended. ", { bold: true }),
    t(
      "A quantified effect from a cited study and a model-propagated consequence are different epistemic objects, and the output labels them as such throughout."
    ),
  ])
);
children.push(
  bullet([
    t("Confidence is capped by data completeness. ", { bold: true }),
    t(
      "Strong literature about a barely-described site does not justify a confident recommendation."
    ),
  ])
);

children.push(h2("Retrieval"));
children.push(...codeBlock([
  "score = 0.45 · dense(query, card)            semantic, min-max normalised",
  "      + 0.30 · bm25(query, card)             rare terms: sodicity, zai, Faidherbia",
  "      + 0.25 · condition_match(site, card)   ← what generic RAG does not have",
  "      × 0.15  if any applies_when predicate is VIOLATED",
]));
children.push(
  body([
    t("The third signal is the important one. "),
    t("“Biodiversity is declining on my land”", { italics: true }),
    t(
      " is roughly equally similar to a Sahel agroforestry card and a temperate nitrogen-deposition card; only the site profile breaks the tie. Condition matching is "
    ),
    t("three-valued", { bold: true }),
    t(
      " — satisfied / violated / unknown — and unknown gets half credit, because absence of information about a site is not evidence against a card. Contradicted evidence is penalised and partitioned rather than silently dropped, because "
    ),
    t(
      "“this popular measure does not apply to you, because your rainfall is below 350 mm”",
      { italics: true }
    ),
    t(" is frequently the most decision-relevant output."),
  ])
);
children.push(
  body([
    t("Measured relevance", { bold: true }),
    t(
      " on a 12-query expert-labelled set: recall@1 = 67%, recall@3 = 92%, held as a regression floor by "
    ),
    mono("tests/test_retrieval.py::test_relevance_baseline"),
    t(
      ". Backends are swappable: the default is a dependency-free persistent numpy store with exact cosine plus a deterministic hashed-TF-IDF embedder (at 73 cards exact search is faster "
    ),
    t("and", { italics: true }),
    t(" more accurate than ANN); "),
    mono("BIOAI_VECTOR_BACKEND=chroma"),
    t(" and "),
    mono("BIOAI_EMBEDDER=minilm"),
    t(" swap in ChromaDB and sentence-transformers."),
  ])
);

children.push(h2("Conversational intelligence"));
children.push(
  body(
    "Clarifying questions are chosen by counterfactual simulation, not by which field happens to be empty. For each unknown field the engine imputes plausible low and high values, re-runs diagnosis and candidate generation, and measures how far the recommendation set moved:"
  )
);
children.push(
  ...codeBlock(["gain = 0.55·mean_divergence + 0.25·spread + 0.20·structural_prior"])
);
children.push(
  body([
    t("The "),
    t("spread", { bold: true }),
    t(
      " term is the real signal: if the low case and the high case lead to the same advice, the answer decides nothing and the question is not worth a turn. This runs entirely on the deterministic layer, so it costs about 50 ms and runs on every turn. A land-use relevance gate also prevents absurd-but-high-scoring questions, such as asking a wheat farmer for a stocking rate."
    ),
  ])
);
children.push(
  ...codeBlock([
    "You > Biodiversity is declining on my land",
    "",
    "1. What is the soil pH?                                    [gain 0.67]",
    "   Why it changes the answer: pH gates whether legume-based nitrogen",
    "   and carbon measures can work at all, so it changes the ORDER OF",
    "   OPERATIONS rather than just the choice.",
    "2. What is the soil organic carbon in the topsoil?         [gain 0.62]",
    "3. What is the average annual rainfall?                    [gain 0.62]",
    "   Why: rainfall decides whether added tree cover raises or lowers",
    "   water availability, which flips the SIGN of the agroforestry",
    "   recommendation.",
  ])
);
children.push(
  body([
    t("Memory is the structured profile, not the transcript", { bold: true }),
    t(
      ", so turn 7 reasons over a number given in turn 2 without re-reading anything, and "
    ),
    t("“actually, the pH is 6.4”", { italics: true }),
    t(
      " overwrites the slot rather than leaving two contradictory claims in a context window. Inferred values never overwrite user-stated ones — provenance is tracked and labelled in the output. The ask-or-advise policy is deliberately not a hard gate: it never asks more than twice in a row, and stops immediately on "
    ),
    t("“just tell me”", { italics: true }),
    t(" or "),
    t("“I don't know”", { italics: true }),
    t(", so it cannot get stuck interrogating a user who does not have the numbers."),
  ])
);

// ---- 4. knowledge base / schema ----
children.push(h1("4. Knowledge base and schema"));
children.push(
  body([
    t(
      "There is no relational database. The knowledge base is version-controlled, human-readable data files under "
    ),
    mono("src/bioai/knowledge/"),
    t(
      ", validated strictly at load time; the only mutable runtime state is short-lived conversational memory. This is a deliberate choice: the scientific content is the asset, so it belongs in code review and git history, where a change to an effect size shows up as a reviewable diff."
    ),
  ])
);

children.push(h2("Contents"));
children.push(
  table(
    ["Artefact", "Size", "Role"],
    [
      [[mono("corpus/*.yaml")], "73 cards", "Evidence cards: quantified, citable, conditional findings"],
      [[mono("metric_bands.csv")], "73 rows", "Interpretation layer: turns 0.3% into “critical” with a sourced reason"],
      [[mono("causal_graph.yaml")], "45 edges", "Signed, weighted, lagged metric interactions; every edge sourced"],
      [[mono("interventions.yaml")], "41 actions", "Catalogue with predicates, cost, monitoring, novelty"],
      [[mono("richness_baselines.csv")], "22 rows", "Expected richness by land use × climate zone × taxon"],
      [[mono("zone_reference.csv")], "6 zones", "Agro-ecological zones for context inference"],
    ],
    [2900, 1400, 5780]
  ),
  spacer(200)
);

children.push(h2("Evidence card schema"));
children.push(
  body(
    "A card is not a text chunk — it is a structured finding. The mechanism field is what separates this from a document store: a retrieved chunk can say cover crops raise soil carbon, but a mechanism says why legume mixes sit at the top of the range, and that is the part which transfers to a site the study never covered."
  )
);
children.push(
  ...codeBlock([
    "- id: ev-cover-crop-soc                    unique, kebab-case",
    "  claim: >                                 MUST carry numbers",
    "    Cover crops raise topsoil organic carbon by roughly 0.20-0.60",
    "    t C/ha/yr (about +8% to +25% relative SOC stock over 5-10 years)...",
    "  mechanism: >                             WHY, not a restatement",
    "    Continuous living roots deliver labile carbon as rhizodeposits into",
    "    the aggregate-forming zone... Legumes lift the N supply, lowering",
    "    residue C:N and raising microbial carbon-use efficiency, so more",
    "    fixed carbon is stabilised as necromass instead of lost as CO2.",
    "  effects:                                 machine-readable, composable",
    "    - metric: soil_organic_carbon_pct",
    "      direction: increase",
    "      relative_change_pct: [8, 25]",
    "      time_to_effect_years: [2.0, 5.0]",
    "  applies_when:                            the applicability envelope",
    "    - field: annual_rainfall_mm",
    "      min: 300",
    "      note: \"below ~300 mm the cover crop competes for water\"",
    "  caveats:                                 become the stated risks",
    "    - \"Termination by tillage can re-mineralise the accumulated carbon.\"",
    "  citation:",
    "    authors: \"Poeplau, C. & Don, A.\"",
    "    year: 2015",
    "    venue: \"Agriculture, Ecosystems & Environment 200:33-41\"",
    "    doi: \"10.1016/j.agee.2014.10.024\"",
    "    type: meta_analysis                    drives the confidence calculus",
  ])
);

children.push(h2("Citation integrity — verified, not asserted"));
children.push(
  body([
    t("The corpus claims real published sources, and "),
    mono("scripts/verify_citations.py"),
    t(
      " lets a reviewer check that claim without taking it on trust. It resolves every DOI against Crossref, falling back to doi.org content negotiation for DataCite registrations such as Zenodo, and compares registered titles against the corpus:"
    ),
  ])
);
children.push(
  ...codeBlock([
    "$ python scripts/verify_citations.py --check-urls",
    "",
    "64/64 DOIs resolved with a matching title, 0 mismatches, 0 unresolvable.",
    "7/7 institutional report URLs live (FAO, IPCC, IPBES, UNCCD, UNEP, IFPRI).",
  ])
);
children.push(
  body(
    "Corpus composition: 20 meta-analyses, 18 systematic reviews, 15 field studies, 10 global assessments (IPCC SRCCL, IPCC AR6 WGIII Ch.7, IPBES 2019, FAO SWSR, UNCCD GLO), 6 long-term experiments, 3 models, 1 technical guideline. 66% is meta-analysis, systematic review or global assessment, which is what justifies the confidence levels the system reports."
  )
);

children.push(h2("Validation is strict and fails loudly"));
children.push(
  body(
    "A silently mis-parsed evidence card would produce a confidently wrong recommendation, so the following prevent the application from starting at all:"
  )
);
children.push(bullet("An evidence card citing an intervention that does not exist."));
children.push(bullet("A causal edge with no evidence, or citing a card that does not exist."));
children.push(bullet("An intervention with no supporting evidence card — nothing is recommendable that cannot be cited."));
children.push(bullet("A duplicate card or intervention ID, a non-contiguous metric band, or a non-monotonic edge with no declared optimum."));
children.push(
  body(
    "Three of these fired for real during development and caught genuine errors in the corpus."
  )
);

children.push(h2("Site profile schema (the input contract)"));
children.push(
  body([
    t("~45 optional fields across soil, climate, water, land-use structure, biodiversity observations, human pressure and geo context, defined as a Pydantic model in "),
    mono("src/bioai/schemas.py"),
    t(
      ". Every field is optional, because incompleteness is the normal case and is precisely what drives the clarifying questions. A "
    ),
    mono("provenance"),
    t(
      " map records whether each value came from the user, was inferred, or is a regional default, and the output labels it accordingly."
    ),
  ])
);

// ---- 5. local setup ----
children.push(h1("5. Local setup"));
children.push(
  body([
    t("Requires Python 3.10+ only. "),
    t("No API key, no database, no network access, no build step.", { bold: true }),
  ])
);
children.push(
  ...codeBlock([
    "git clone " + REPO_URL,
    "cd darukaa-biodiversity-engine",
    "python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate",
    "pip install -r requirements.txt",
    "",
    "# The brief's worked example, end to end:",
    "PYTHONPATH=src python -m bioai.cli demo 1",
  ])
);

children.push(h2("Entry points"));
children.push(
  table(
    ["Command", "Purpose"],
    [
      [[mono("python -m bioai.cli chat")], "Interactive multi-turn session"],
      [[mono("python -m bioai.cli demo 1..5")], "Five scripted scenarios across contrasting biomes"],
      [[mono("python -m bioai.cli assess --json examples/...")], "One-shot structured assessment (6 example profiles provided)"],
      [[mono("python -m bioai.cli search \"...\"")], "Inspect the retrieval layer with per-signal scores"],
      [[mono("python -m bioai.cli stats")], "Knowledge base composition"],
      [[mono("uvicorn bioai.api.app:app --reload")], "API on :8000, web client at /, OpenAPI at /docs"],
      [[mono("pytest tests/ -q")], "116 tests, ~14 s, entirely offline"],
    ],
    [4400, 5680]
  ),
  spacer(200)
);

children.push(h2("Enabling the optional LLM layer"));
children.push(
  body([
    mono("cp .env.example .env"),
    t(" and set "),
    mono("ANTHROPIC_API_KEY"),
    t(
      ". Nothing else changes. This adds natural-language parsing for phrasings the deterministic extractor misses, and narration of the engine's output; "
    ),
    mono("/healthz"),
    t(
      " reports which mode is active. The narration prompt is constrained so the model may not add, alter or drop any number, citation, confidence level, horizon or risk, and the verbatim engine output is always retained beneath the prose."
    ),
  ])
);

children.push(h2("API surface"));
children.push(
  table(
    ["Endpoint", "Purpose"],
    [
      [[mono("POST /chat")], "Multi-turn conversation; free text and/or structured profile"],
      [[mono("POST /assess")], "One-shot structured assessment; any subset of fields is valid"],
      [[mono("GET /session/{id}")], "Inspect accumulated memory, transcript, fields asked vs answered"],
      [[mono("POST /knowledge/search")], "Retrieval with per-signal scores and site-condition verdicts"],
      [[mono("GET /knowledge/graph")], "The causal graph with mechanisms and sources"],
      [[mono("GET /knowledge/stats")], "Corpus composition and full citation list"],
      [[mono("GET /healthz")], "Knowledge base size, retrieval backend, LLM status"],
    ],
    [3000, 7080]
  ),
  spacer(160)
);
children.push(
  body([
    t("Passing "),
    mono("explain_retrieval: true"),
    t(
      " returns the fusion weights, per-signal scores, matched and violated site predicates, and the complete scoring trace for every recommendation — so the reasoning is auditable from outside the process, not merely described."
    ),
  ])
);

// ---- 6. CI/CD ----
children.push(h1("6. CI/CD"));
children.push(
  body([
    mono(".github/workflows/ci.yml"),
    t(" runs on every push and pull request, in three jobs."),
  ])
);
children.push(
  bullet([
    t("test", { bold: true }),
    t(" — matrix across Python 3.10 / 3.11 / 3.12, with "),
    mono("BIOAI_LLM_ENABLED=false"),
    t(
      " set deliberately so CI proves the deterministic engine carries the entire suite on its own. It validates the knowledge base and builds the retrieval index (so a dangling reference or unsourced causal edge fails loudly before any test runs), runs all 116 tests, then smoke-tests every example profile and all five scripted scenarios end to end."
    ),
  ])
);
children.push(
  bullet([
    t("lint", { bold: true }),
    t(" — "),
    mono("ruff check"),
    t(" over "),
    mono("src"),
    t(", "),
    mono("tests"),
    t(" and "),
    mono("scripts"),
    t(", currently clean."),
  ])
);
children.push(
  bullet([
    t("docker", { bold: true }),
    t(
      " — builds the image, starts the container with no API key, waits for health, then asserts from outside the process that the knowledge base loaded (≥60 cards, ≥40 edges, vectors indexed = cards) and that a full assessment is produced in which "
    ),
    t("every recommendation carries a citation and touches at least three distinct metrics", { bold: true }),
    t(" — the brief's own constraints, enforced as a build gate."),
  ])
);

children.push(h2("Deployment"));
children.push(
  table(
    ["Target", "How"],
    [
      ["Docker", [mono("docker build -t bioai . && docker run -p 8000:8000 bioai")]],
      ["Render", [mono("render.yaml"), t(" blueprint — free plan, health check "), mono("/healthz"), t(", no secrets required")]],
      ["Railway / Heroku", [mono("Procfile")]],
      ["Fly.io / Cloud Run", [t("Same image; it honours "), mono("$PORT")]],
    ],
    [2500, 7580]
  ),
  spacer(160)
);
children.push(
  body(
    "The retrieval index is built at image-build time, so the container starts warm and a malformed evidence card fails the build rather than a user request. Because no secret is required for full functionality, the deployment has no configuration step at all."
  )
);

// ---- 7. testing ----
children.push(h1("7. Testing"));
children.push(
  table(
    ["Suite", "What it protects"],
    [
      [[mono("test_knowledge.py"), t(" (15)")], "Every claim quantified and cited; every causal edge sourced; every intervention citeable; DOIs well-formed; bands contiguous; corpus weighted toward strong evidence"],
      [[mono("test_retrieval.py"), t(" (19)")], "Measured relevance floor; site-conditioning changes ranking; contradicted evidence excluded from support but retrievable; index persistence and staleness detection"],
      [[mono("test_reasoning.py"), t(" (34)")], "Multi-metric diagnoses; opposite sites get opposite advice; non-monotonic edges flip sign with site value; saturation attenuates propagation; contraindications exclude inapplicable measures; ≥3 metrics per recommendation; confidence capped by completeness; determinism"],
      [[mono("test_dialogue.py"), t(" (33)")], "Unit conversion, negation, implausible-value rejection; questions ranked by information gain and adapted to the site; memory accumulates and corrections overwrite; asks, then advises, then stops asking"],
      [[mono("test_api.py"), t(" (15)")], "HTTP contracts, session lifecycle, validation, retrieval introspection"],
    ],
    [2500, 7580]
  ),
  spacer(200)
);
children.push(
  body(
    "Several of these caught real defects during development rather than merely documenting intent: a falsy-zero session TTL default that disabled expiry; graph saturation being read off the target metric instead of the source, so habitat-rich and cleared landscapes were given identical predicted gains; a negation losing to a keyword match, so “no invasive weeds” was recorded as invasives present; a habitat vocabulary too narrow to recognise “scrub”, which silently dropped an isolation distance; and an include_violated flag that did nothing because penalised cards fell below the result cut."
  )
);

// ---- 8. limitations ----
children.push(h1("8. Known limitations"));
children.push(
  body(
    "Stated plainly, because a system that reports on scientific uncertainty should be honest about its own."
  )
);
children.push(
  bullet([
    t("Effect-size ranges are literature-derived approximations", { bold: true }),
    t(
      ", not site-specific predictions. They are the right order of magnitude for the conditions the cited studies cover; they are not a substitute for local trials."
    ),
  ])
);
children.push(
  bullet([
    t("The causal graph is hand-built.", { bold: true }),
    t(
      " Edge strengths are expert-assigned weights informed by the cited work, not fitted coefficients. They encode direction, relative importance and lag defensibly; they are not calibrated for quantitative prediction."
    ),
  ])
);
children.push(
  bullet([
    t("Secondary effect magnitudes are propagation strengths", { bold: true }),
    t(", not measured quantities — labelled as such wherever they appear."),
  ])
);
children.push(
  bullet([
    t("Geo-coordinates are context, not a data source.", { bold: true }),
    t(
      " There is no live SoilGrids or remote-sensing lookup; coordinates are recorded and used for regional context only. Wiring a soil-grid or rainfall raster in behind "
    ),
    mono("metrics.infer_context()"),
    t(" is the obvious next step and the interface is already shaped for it."),
  ])
);
children.push(
  bullet([
    t("The corpus is global-generalist", { bold: true }),
    t(
      ", weighted toward dryland agro-ecosystems, temperate arable and tropical restoration, because that is where the strongest quantitative meta-analytic evidence sits. Boreal, alpine and freshwater systems are thin."
    ),
  ])
);
children.push(
  bullet([
    t("Sessions are in-memory", { bold: true }),
    t(" — deliberate for short-lived conversational state, but horizontal scaling needs the "),
    mono("SessionStore"),
    t(" interface pointed at Redis. It is a single-class change."),
  ])
);

// ---- 9. notes for reviewers ----
children.push(h1("9. Notes for reviewers"));
children.push(
  bullet([
    t("Fastest path to the substance: ", { bold: true }),
    mono("PYTHONPATH=src python -m bioai.cli demo 1"),
    t(" then "),
    mono("demo 2"),
    t(
      ". Those two scenarios are deliberate mirror images and show the engine giving opposite, correctly-reasoned prescriptions to superficially similar problems."
    ),
  ])
);
children.push(
  bullet([
    t("To see the reasoning rather than the conclusion: ", { bold: true }),
    mono("demo 1 --explain"),
    t(" prints the full scoring trace, or pass "),
    mono("explain_retrieval: true"),
    t(" to "),
    mono("POST /assess"),
    t("."),
  ])
);
children.push(
  bullet([
    t("To audit the knowledge layer independently: ", { bold: true }),
    t("the "),
    mono("/knowledge/*"),
    t(" endpoints and the "),
    mono("Knowledge retrieval"),
    t(
      " tab of the web client expose retrieval with per-signal scores and the satisfied / violated / unknown site predicates for each card."
    ),
  ])
);
children.push(
  bullet([
    t("To check the science is real: ", { bold: true }),
    mono("python scripts/verify_citations.py --check-urls"),
    t(" (needs network). Every DOI resolves with a matching title."),
  ])
);
children.push(
  bullet([
    t("Documentation: ", { bold: true }),
    mono("README.md"),
    t(" for the overview, "),
    mono("docs/ARCHITECTURE.md"),
    t(" for the full pipeline, scoring formulas and failure behaviour, and "),
    mono("docs/KNOWLEDGE_SCHEMA.md"),
    t(
      " for the data contract — written so a domain scientist can extend the corpus without touching Python."
    ),
  ])
);

children.push(
  new Paragraph({
    spacing: { before: 420 },
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 8 } },
    children: [
      new TextRun({
        text: "73 evidence cards · 71 citations, 64/64 DOIs verified · 45 sourced causal edges · 41 interventions · 27 diagnostic rules · 116 tests · runs offline with no API key",
        font: "Calibri",
        size: 19,
        color: MUTED,
        italics: true,
      }),
    ],
  })
);

// ---------------------------------------------------------------------------
// assemble
// ---------------------------------------------------------------------------

const doc = new Document({
  creator: "Khushi",
  title: "Darukaa.Earth Submission — Biodiversity Intelligence Engine",
  description: "Hackathon submission document",
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 360, hanging: 220 } } },
          },
          {
            level: 1,
            format: LevelFormat.BULLET,
            text: "◦",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 220 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE.width, height: PAGE.height },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        },
      },
      children,
    },
  ],
});

const out = path.join(__dirname, "Darukaa_Submission_Biodiversity_Intelligence_Engine.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${(buf.length / 1024).toFixed(0)} KB)`);
});
