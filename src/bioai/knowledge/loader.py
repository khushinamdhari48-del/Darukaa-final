"""Loads and validates the knowledge base: evidence corpus (YAML), structured
reference datasets (CSV) and the causal graph / intervention catalogue (YAML).

Validation is strict and fails loudly at load time, because a silently
mis-parsed evidence card would produce a confidently wrong recommendation.
"""
from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import settings
from ..schemas import Condition, EvidenceCard

# ---------------------------------------------------------------------------
# Structured reference data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricBand:
    metric: str
    band: str
    low: float
    high: float
    verdict: str  # critical | poor | moderate | good | high
    interpretation: str
    source: str

    def contains(self, value: float) -> bool:
        # Half-open interval so adjacent bands do not both claim a boundary value,
        # except for the topmost band of each metric which is closed.
        return self.low <= value < self.high


@dataclass(frozen=True)
class ZoneReference:
    climate_zone: str
    rainfall_min_mm: float
    rainfall_max_mm: float
    typical_dry_season_months: int
    aridity_index_range: str
    notes: str
    source: str


@dataclass(frozen=True)
class RichnessBaseline:
    land_use: str
    climate_zone: str
    taxon: str
    expected_min: float
    expected_max: float
    unit: str
    notes: str
    source: str


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    sign: str  # positive | negative | non_monotonic
    strength: float
    lag_years: float
    mechanism: str
    evidence: tuple[str, ...]
    optimum: float | None = None
    saturating_at: float | None = None
    threshold: float | None = None

    @property
    def horizon(self) -> str:
        if self.lag_years <= 2:
            return "short"
        if self.lag_years <= 5:
            return "medium"
        return "long"


@dataclass(frozen=True)
class Intervention:
    id: str
    title: str
    what_to_do: str
    targets: tuple[str, ...]
    addresses: tuple[str, ...]
    requires: tuple[Condition, ...]
    contraindications: tuple[Condition, ...]
    cost_intensity: str
    reversibility: str
    time_horizon: str
    establishment_years: tuple[float, float]
    monitoring: tuple[str, ...]
    synergies: tuple[str, ...]
    conflicts: tuple[str, ...]
    novelty: str


@dataclass
class KnowledgeBase:
    cards: list[EvidenceCard]
    bands: list[MetricBand]
    zones: list[ZoneReference]
    richness: list[RichnessBaseline]
    edges: list[CausalEdge]
    interventions: list[Intervention]

    # -- indexes ------------------------------------------------------------
    def __post_init__(self) -> None:
        self.cards_by_id = {c.id: c for c in self.cards}
        self.interventions_by_id = {i.id: i for i in self.interventions}
        self.cards_by_intervention: dict[str, list[EvidenceCard]] = {}
        for card in self.cards:
            for iv in card.interventions:
                self.cards_by_intervention.setdefault(iv, []).append(card)
        self.edges_from: dict[str, list[CausalEdge]] = {}
        self.edges_to: dict[str, list[CausalEdge]] = {}
        for edge in self.edges:
            self.edges_from.setdefault(edge.source, []).append(edge)
            self.edges_to.setdefault(edge.target, []).append(edge)
        self.bands_by_metric: dict[str, list[MetricBand]] = {}
        for band in self.bands:
            self.bands_by_metric.setdefault(band.metric, []).append(band)
        for metric_bands in self.bands_by_metric.values():
            metric_bands.sort(key=lambda b: b.low)

    # -- lookups ------------------------------------------------------------
    def band_for(self, metric: str, value: float | None) -> MetricBand | None:
        if value is None:
            return None
        metric_bands = self.bands_by_metric.get(metric)
        if not metric_bands:
            return None
        for band in metric_bands:
            if band.contains(value):
                return band
        # value at or above the top of the last band
        last = metric_bands[-1]
        if value >= last.low:
            return last
        return None

    def zone_for_rainfall(self, rainfall_mm: float | None) -> ZoneReference | None:
        if rainfall_mm is None:
            return None
        # `montane` is temperature-defined, not rainfall-defined, so it is never
        # inferred from rainfall alone.
        candidates = [z for z in self.zones if z.climate_zone != "montane"]
        for zone in candidates:
            if zone.rainfall_min_mm <= rainfall_mm < zone.rainfall_max_mm:
                return zone
        return candidates[-1] if rainfall_mm >= candidates[-1].rainfall_min_mm else None

    def richness_baseline(
        self, land_use: str | None, climate_zone: str | None, taxon: str = "vascular_plants"
    ) -> RichnessBaseline | None:
        if not land_use:
            return None
        exact = [
            r
            for r in self.richness
            if r.land_use == land_use and r.taxon == taxon and r.climate_zone == climate_zone
        ]
        if exact:
            return exact[0]
        # fall back to the same land use in any zone, flagged by the caller as a
        # weaker comparison
        loose = [r for r in self.richness if r.land_use == land_use and r.taxon == taxon]
        return loose[0] if loose else None

    def evidence_for_intervention(self, intervention_id: str) -> list[EvidenceCard]:
        return self.cards_by_intervention.get(intervention_id, [])

    def stats(self) -> dict[str, int]:
        return {
            "evidence_cards": len(self.cards),
            "distinct_citations": len({c.citation.formatted() for c in self.cards}),
            "metric_bands": len(self.bands),
            "causal_edges": len(self.edges),
            "interventions": len(self.interventions),
            "richness_baselines": len(self.richness),
            "climate_zones": len(self.zones),
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_conditions(raw: Any, where: str) -> tuple[Condition, ...]:
    out: list[Condition] = []
    for item in raw or []:
        if not isinstance(item, dict) or "field" not in item:
            raise ValueError(f"{where}: condition must be a mapping with a 'field' key, got {item!r}")
        out.append(Condition(**item))
    return tuple(out)


def _load_corpus(corpus_dir: Path) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    seen: set[str] = set()
    files = sorted(corpus_dir.glob("*.yaml")) + sorted(corpus_dir.glob("*.yml"))
    if not files:
        raise FileNotFoundError(f"no evidence corpus files found in {corpus_dir}")
    for path in files:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path.name}: expected a top-level list of evidence cards")
        for entry in raw:
            card = EvidenceCard(**entry)
            if card.id in seen:
                raise ValueError(f"duplicate evidence card id {card.id!r} in {path.name}")
            seen.add(card.id)
            cards.append(card)
    return cards


def _load_bands(path: Path) -> list[MetricBand]:
    out: list[MetricBand] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                MetricBand(
                    metric=row["metric"],
                    band=row["band"],
                    low=float(row["low"]),
                    high=float(row["high"]),
                    verdict=row["verdict"],
                    interpretation=row["interpretation"],
                    source=row["source"],
                )
            )
    return out


def _load_zones(path: Path) -> list[ZoneReference]:
    out: list[ZoneReference] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                ZoneReference(
                    climate_zone=row["climate_zone"],
                    rainfall_min_mm=float(row["rainfall_min_mm"]),
                    rainfall_max_mm=float(row["rainfall_max_mm"]),
                    typical_dry_season_months=int(row["typical_dry_season_months"]),
                    aridity_index_range=row["aridity_index_range"],
                    notes=row["notes"],
                    source=row["source"],
                )
            )
    out.sort(key=lambda z: z.rainfall_min_mm)
    return out


def _load_richness(path: Path) -> list[RichnessBaseline]:
    out: list[RichnessBaseline] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                RichnessBaseline(
                    land_use=row["land_use"],
                    climate_zone=row["climate_zone"],
                    taxon=row["taxon"],
                    expected_min=float(row["expected_min"]),
                    expected_max=float(row["expected_max"]),
                    unit=row["unit"],
                    notes=row["notes"],
                    source=row["source"],
                )
            )
    return out


def _load_edges(path: Path, valid_card_ids: set[str]) -> list[CausalEdge]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[CausalEdge] = []
    for entry in raw.get("edges", []):
        evidence = tuple(entry.get("evidence", []))
        unknown = [e for e in evidence if e not in valid_card_ids]
        if unknown:
            raise ValueError(
                f"causal edge {entry.get('source')} -> {entry.get('target')} cites "
                f"unknown evidence card(s): {unknown}"
            )
        if not evidence:
            raise ValueError(
                f"causal edge {entry.get('source')} -> {entry.get('target')} has no evidence; "
                "every edge must be sourced"
            )
        out.append(
            CausalEdge(
                source=entry["source"],
                target=entry["target"],
                sign=entry["sign"],
                strength=float(entry["strength"]),
                lag_years=float(entry["lag_years"]),
                mechanism=entry["mechanism"].strip(),
                evidence=evidence,
                optimum=entry.get("optimum"),
                saturating_at=entry.get("saturating_at"),
                threshold=entry.get("threshold"),
            )
        )
    return out


def _load_interventions(path: Path) -> list[Intervention]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[Intervention] = []
    seen: set[str] = set()
    for entry in raw.get("interventions", []):
        iid = entry["id"]
        if iid in seen:
            raise ValueError(f"duplicate intervention id {iid!r}")
        seen.add(iid)
        est = entry.get("establishment_years", [1.0, 3.0])
        out.append(
            Intervention(
                id=iid,
                title=entry["title"],
                what_to_do=entry["what_to_do"].strip(),
                targets=tuple(entry.get("targets", [])),
                addresses=tuple(entry.get("addresses", [])),
                requires=_parse_conditions(entry.get("requires"), f"intervention {iid} requires"),
                contraindications=_parse_conditions(
                    entry.get("contraindications"), f"intervention {iid} contraindications"
                ),
                cost_intensity=entry.get("cost_intensity", "moderate"),
                reversibility=entry.get("reversibility", "moderate"),
                time_horizon=entry.get("time_horizon", "medium"),
                establishment_years=(float(est[0]), float(est[1])),
                monitoring=tuple(entry.get("monitoring", [])),
                synergies=tuple(entry.get("synergies", [])),
                conflicts=tuple(entry.get("conflicts", [])),
                novelty=entry.get("novelty", "standard"),
            )
        )
    return out


def _cross_validate(kb: KnowledgeBase) -> None:
    """Referential integrity across the knowledge base."""
    problems: list[str] = []

    iv_ids = set(kb.interventions_by_id)
    for card in kb.cards:
        for iv in card.interventions:
            if iv not in iv_ids:
                problems.append(f"evidence card {card.id} references unknown intervention {iv!r}")

    for iv in kb.interventions:
        for other in (*iv.synergies, *iv.conflicts):
            if other not in iv_ids:
                problems.append(f"intervention {iv.id} references unknown intervention {other!r}")
        if not kb.evidence_for_intervention(iv.id):
            problems.append(
                f"intervention {iv.id} has no supporting evidence card; it could never be "
                "cited and must not be recommendable"
            )

    if problems:
        raise ValueError("knowledge base validation failed:\n  - " + "\n  - ".join(problems))


@functools.lru_cache(maxsize=1)
def load_knowledge_base() -> KnowledgeBase:
    """Load, validate and cache the whole knowledge base."""
    cards = _load_corpus(settings.corpus_dir)
    card_ids = {c.id for c in cards}
    kb = KnowledgeBase(
        cards=cards,
        bands=_load_bands(settings.dataset_dir / "metric_bands.csv"),
        zones=_load_zones(settings.dataset_dir / "zone_reference.csv"),
        richness=_load_richness(settings.dataset_dir / "richness_baselines.csv"),
        edges=_load_edges(settings.dataset_dir / "causal_graph.yaml", card_ids),
        interventions=_load_interventions(settings.dataset_dir / "interventions.yaml"),
    )
    _cross_validate(kb)
    return kb
