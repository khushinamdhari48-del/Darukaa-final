"""Typed contracts for the whole system.

Everything that crosses a module boundary is one of these models. The reasoning
engine never sees free text, and the dialogue layer never sees raw dicts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class LandUse(str, Enum):
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    PASTURE = "pasture"
    ORCHARD = "orchard"
    PLANTATION = "plantation"
    AGROFORESTRY = "agroforestry"
    FOREST = "forest"
    DEGRADED = "degraded_barren"
    WETLAND = "wetland"
    MANGROVE = "mangrove"
    PERI_URBAN = "peri_urban"


class CroppingSystem(str, Enum):
    MONOCULTURE = "monoculture"
    ROTATION = "rotation"
    INTERCROP = "intercrop"
    AGROFORESTRY = "agroforestry"
    FALLOW = "fallow"
    NOT_APPLICABLE = "not_applicable"


class RainfallPattern(str, Enum):
    UNIMODAL = "unimodal"
    BIMODAL = "bimodal"
    ERRATIC = "erratic"
    ASEASONAL = "aseasonal"


class ClimateZone(str, Enum):
    ARID = "arid"
    SEMI_ARID = "semi_arid"
    DRY_SUBHUMID = "dry_subhumid"
    SUBHUMID = "subhumid"
    HUMID = "humid"
    MONTANE = "montane"


class TimeHorizon(str, Enum):
    SHORT = "short"      # <= 2 years to measurable effect
    MEDIUM = "medium"    # 2-5 years
    LONG = "long"        # > 5 years


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"


class EvidenceType(str, Enum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    LONG_TERM_EXPERIMENT = "long_term_experiment"
    FIELD_STUDY = "field_study"
    MODEL = "model"
    GLOBAL_ASSESSMENT = "global_assessment"
    TECHNICAL_GUIDELINE = "technical_guideline"
    DATASET = "dataset"


# Ordered by inferential strength; drives the confidence calculus.
EVIDENCE_WEIGHT: dict[str, float] = {
    EvidenceType.META_ANALYSIS.value: 1.00,
    EvidenceType.SYSTEMATIC_REVIEW.value: 0.95,
    EvidenceType.GLOBAL_ASSESSMENT.value: 0.90,
    EvidenceType.LONG_TERM_EXPERIMENT.value: 0.85,
    EvidenceType.MODEL.value: 0.70,
    EvidenceType.FIELD_STUDY.value: 0.65,
    EvidenceType.TECHNICAL_GUIDELINE.value: 0.60,
    EvidenceType.DATASET.value: 0.55,
}


# ---------------------------------------------------------------------------
# Site state
# ---------------------------------------------------------------------------


class GeoContext(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None
    region_name: str | None = None
    country: str | None = None
    agro_ecological_zone: str | None = None

    @field_validator("latitude")
    @classmethod
    def _lat(cls, v: float | None) -> float | None:
        if v is not None and not -90 <= v <= 90:
            raise ValueError("latitude must be within [-90, 90]")
        return v

    @field_validator("longitude")
    @classmethod
    def _lon(cls, v: float | None) -> float | None:
        if v is not None and not -180 <= v <= 180:
            raise ValueError("longitude must be within [-180, 180]")
        return v

    @property
    def has_point(self) -> bool:
        return self.latitude is not None and self.longitude is not None


class SiteProfile(BaseModel):
    """Every field is optional. Incompleteness is the normal case and is what
    drives clarifying questions; `provenance` records how each value arrived
    (user / inferred / regional_default) so the engine can discount inferences.
    """

    # --- soil ---
    soil_organic_carbon_pct: float | None = Field(None, ge=0, le=60)
    soil_ph: float | None = Field(None, ge=2, le=11)
    soil_moisture_pct: float | None = Field(None, ge=0, le=100, description="volumetric %")
    soil_bulk_density_g_cm3: float | None = Field(None, ge=0.5, le=2.2)
    soil_texture: Literal["sand", "sandy_loam", "loam", "silt_loam", "clay_loam", "clay"] | None = None
    soil_nitrogen_pct: float | None = Field(None, ge=0, le=5)
    salinity_ec_ds_m: float | None = Field(None, ge=0, le=60)
    erosion_class: Literal["none", "slight", "moderate", "severe"] | None = None
    slope_pct: float | None = Field(None, ge=0, le=100)

    # --- climate / water ---
    annual_rainfall_mm: float | None = Field(None, ge=0, le=12000)
    rainfall_pattern: RainfallPattern | None = None
    dry_season_months: int | None = Field(None, ge=0, le=12)
    mean_annual_temp_c: float | None = Field(None, ge=-20, le=45)
    climate_zone: ClimateZone | None = None
    water_table_depth_m: float | None = Field(None, ge=0, le=500)
    irrigation: Literal["none", "rainfed_supplemental", "flood", "furrow", "sprinkler", "drip"] | None = None

    # --- land use / structure ---
    land_use: LandUse | None = None
    cropping_system: CroppingSystem | None = None
    primary_crop: str | None = None
    area_ha: float | None = Field(None, gt=0)
    tree_cover_pct: float | None = Field(None, ge=0, le=100)
    native_vegetation_pct: float | None = Field(None, ge=0, le=100)
    mean_patch_size_ha: float | None = Field(None, gt=0)
    hedgerow_density_m_per_ha: float | None = Field(None, ge=0)
    distance_to_natural_habitat_m: float | None = Field(None, ge=0)
    permanent_water_present: bool | None = None

    # --- biodiversity observations ---
    species_richness_observed: int | None = Field(None, ge=0)
    bird_species_count: int | None = Field(None, ge=0)
    pollinator_activity: Literal["absent", "low", "moderate", "high"] | None = None
    earthworm_count_per_m2: float | None = Field(None, ge=0)
    invasive_species_present: bool | None = None
    keystone_or_threatened_species: list[str] = Field(default_factory=list)

    # --- human pressure ---
    fertilizer_n_kg_ha_yr: float | None = Field(None, ge=0, le=1000)
    pesticide_applications_per_season: int | None = Field(None, ge=0, le=40)
    tillage: Literal["none", "reduced", "conventional", "deep"] | None = None
    grazing_intensity_lsu_ha: float | None = Field(None, ge=0, le=10)
    residue_removal: Literal["none", "partial", "full", "burned"] | None = None
    deforestation_last_5yr_pct: float | None = Field(None, ge=0, le=100)
    nearby_pollution_source: str | None = None

    # --- context ---
    geo: GeoContext = Field(default_factory=GeoContext)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    notes: str | None = None

    provenance: dict[str, Literal["user", "inferred", "regional_default"]] = Field(default_factory=dict)

    # -- helpers ------------------------------------------------------------
    def known(self) -> dict[str, Any]:
        """Non-null, non-empty environmental fields."""
        out: dict[str, Any] = {}
        for name, value in self.model_dump(exclude={"provenance"}).items():
            if name == "geo":
                if any(v is not None for v in value.values()):
                    out["geo"] = value
                continue
            if value is None or value == [] or value == {}:
                continue
            out[name] = value
        return out

    def get_field(self, name: str) -> Any:
        """Dotted access so conditions can address `geo.latitude`."""
        if "." in name:
            head, tail = name.split(".", 1)
            sub = getattr(self, head, None)
            return getattr(sub, tail, None) if sub is not None else None
        return getattr(self, name, None)

    def merge(self, other: SiteProfile) -> SiteProfile:
        """Right-biased merge used to fold each conversational turn into memory.

        Later user statements override earlier ones, but an inferred value never
        overwrites a value the user stated directly.
        """
        base = self.model_dump()
        incoming = other.model_dump()
        prov = dict(base.get("provenance") or {})
        incoming_prov = incoming.get("provenance") or {}
        for key, value in incoming.items():
            if key == "provenance":
                continue
            if key == "geo":
                merged_geo = dict(base["geo"])
                for gk, gv in value.items():
                    if gv is not None:
                        merged_geo[gk] = gv
                base["geo"] = merged_geo
                continue
            if key in ("objectives", "constraints", "keystone_or_threatened_species"):
                base[key] = list(dict.fromkeys([*(base.get(key) or []), *(value or [])]))
                continue
            if value is None:
                continue
            source = incoming_prov.get(key, "user")
            if prov.get(key) == "user" and source != "user":
                continue
            base[key] = value
            prov[key] = source
        base["provenance"] = prov
        return SiteProfile(**base)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    authors: str
    year: int
    title: str
    venue: str
    doi: str | None = None
    url: str | None = None
    type: EvidenceType

    def _surnames(self) -> list[str]:
        """Surnames from an "Last, F. M., Last, F. M. & Last, F." author string.

        Names arrive as (surname, initials) pairs, so the surnames are the
        even-indexed tokens of each ampersand-separated group. Institutional
        authors ("FAO", "IPCC") have no comma and come back as a single name.
        """
        surnames: list[str] = []
        for group in self.authors.split(" & "):
            tokens = [t.strip() for t in group.split(",") if t.strip()]
            surnames.extend(tokens[::2])
        return surnames or [self.authors.strip()]

    def short(self) -> str:
        names = self._surnames()
        if len(names) == 1:
            return f"{names[0]} {self.year}"
        if len(names) == 2:
            return f"{names[0]} & {names[1]} {self.year}"
        return f"{names[0]} et al. {self.year}"

    def formatted(self) -> str:
        tail = f" doi:{self.doi}" if self.doi else (f" {self.url}" if self.url else "")
        return f"{self.authors} ({self.year}). {self.title}. {self.venue}.{tail}"


class QuantifiedEffect(BaseModel):
    """A single number-carrying claim, kept machine-readable so the reasoning
    layer can compose effects rather than merely quote them."""

    metric: str
    direction: Direction
    absolute_change: str | None = None
    relative_change_pct: tuple[float, float] | None = None
    time_to_effect_years: tuple[float, float] = (1.0, 3.0)
    measurement_context: str | None = None


class Condition(BaseModel):
    """Applicability predicate. Numeric fields take {min,max}; categoricals take
    `allowed` / `forbidden` value lists."""

    field: str
    min: float | None = None
    max: float | None = None
    allowed: list[str] | None = None
    forbidden: list[str] | None = None
    note: str | None = None

    def describe(self) -> str:
        if self.allowed is not None:
            return f"{self.field} in {{{', '.join(self.allowed)}}}"
        if self.forbidden is not None:
            return f"{self.field} not in {{{', '.join(self.forbidden)}}}"
        if self.min is not None and self.max is not None:
            return f"{self.min} <= {self.field} <= {self.max}"
        if self.min is not None:
            return f"{self.field} >= {self.min}"
        if self.max is not None:
            return f"{self.field} <= {self.max}"
        return self.field

    def evaluate(self, profile_value: Any) -> bool | None:
        """True = satisfied, False = violated, None = unknown (cannot judge)."""
        if profile_value is None:
            return None
        if isinstance(profile_value, Enum):
            profile_value = profile_value.value
        if isinstance(profile_value, bool) and (self.allowed or self.forbidden):
            profile_value = "true" if profile_value else "false"
        if self.allowed is not None:
            return str(profile_value) in self.allowed
        if self.forbidden is not None:
            return str(profile_value) not in self.forbidden
        try:
            numeric = float(profile_value)
        except (TypeError, ValueError):
            return None
        if self.min is not None and numeric < self.min:
            return False
        return not (self.max is not None and numeric > self.max)


class EvidenceCard(BaseModel):
    """The atomic unit of the knowledge base: one retrievable, citable,
    quantified finding."""

    id: str
    claim: str
    mechanism: str
    domain: Literal[
        "soil", "biodiversity", "climate", "water", "land_use", "human_impact", "cross_cutting"
    ]
    effects: list[QuantifiedEffect] = Field(default_factory=list)
    applies_when: list[Condition] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    citation: Citation
    interventions: list[str] = Field(default_factory=list, description="intervention ids this card supports")
    tags: list[str] = Field(default_factory=list)
    text: str = Field("", description="prose chunk that is embedded and BM25-indexed")

    def embedding_text(self) -> str:
        parts = [self.claim, self.mechanism, self.text, " ".join(self.tags), self.domain, self.citation.title]
        return "\n".join(p for p in parts if p)


class RetrievedEvidence(BaseModel):
    card: EvidenceCard
    dense_score: float = 0.0
    lexical_score: float = 0.0
    condition_score: float = 0.0
    score: float = 0.0
    satisfied_conditions: list[str] = Field(default_factory=list)
    violated_conditions: list[str] = Field(default_factory=list)
    unknown_conditions: list[str] = Field(default_factory=list)
    retrieval_path: str = "hybrid"


# ---------------------------------------------------------------------------
# Reasoning output
# ---------------------------------------------------------------------------


class MetricImpact(BaseModel):
    metric: str
    direction: Direction
    magnitude: str
    time_to_effect_years: tuple[float, float]
    pathway: str | None = None
    is_secondary: bool = False


class Diagnosis(BaseModel):
    """What the engine believes is wrong, before it proposes anything."""

    code: str
    label: str
    severity: float = Field(ge=0, le=1)
    limiting_factor: str | None = None
    explanation: str
    metrics_implicated: list[str] = Field(default_factory=list)
    interacting_metrics: list[str] = Field(default_factory=list)


class Confidence(BaseModel):
    score: float = Field(ge=0, le=1)
    label: Literal["low", "moderate", "high", "very_high"]
    drivers: list[str] = Field(default_factory=list)
    limiters: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    id: str
    title: str
    what_to_do: str
    why_it_works: str
    causal_pathway: list[str] = Field(default_factory=list)
    impacted_metrics: list[MetricImpact] = Field(default_factory=list)
    time_horizon: TimeHorizon
    confidence: Confidence
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    risks_and_tradeoffs: list[str] = Field(default_factory=list)
    synergies: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    monitoring: list[str] = Field(default_factory=list)
    cost_intensity: Literal["low", "moderate", "high"] = "moderate"
    reversibility: Literal["easy", "moderate", "hard"] = "moderate"
    priority_score: float = 0.0
    novelty: Literal["standard", "non_obvious"] = "standard"
    rationale_trace: list[str] = Field(default_factory=list)


class ClarifyingQuestion(BaseModel):
    field: str
    question: str
    why_it_matters: str
    expected_information_gain: float
    unit_or_options: str | None = None


class SiteAssessment(BaseModel):
    profile: SiteProfile
    derived_indices: dict[str, float]
    diagnoses: list[Diagnosis]
    recommendations: list[Recommendation]
    clarifying_questions: list[ClarifyingQuestion]
    data_completeness: float
    interaction_notes: list[str] = Field(default_factory=list)
    retrieval_debug: list[dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = ""
    profile: SiteProfile | None = None
    max_recommendations: int = Field(4, ge=1, le=10)
    explain_retrieval: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    assessment: SiteAssessment | None = None
    asked_for: list[ClarifyingQuestion] = Field(default_factory=list)
    turn: int = 0
    llm_used: bool = False
