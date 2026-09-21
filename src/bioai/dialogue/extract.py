"""Natural language -> SiteProfile.

Two extractors behind one interface:

  `rule_extract`  - deterministic patterns over units, keywords and qualitative
                    phrasing. Runs always, needs no key, and is the reason the
                    system is usable offline.
  `llm_extract`   - an Anthropic call that returns a strict JSON patch, used when
                    a key is present, for phrasings the patterns miss.

The rule extractor runs first and its results take precedence, because a number
the user wrote explicitly should never be overridden by a model's paraphrase of
it. The LLM only fills gaps.
"""
from __future__ import annotations

import re
from typing import Any

from ..schemas import ClimateZone, CroppingSystem, LandUse, RainfallPattern, SiteProfile

# ---------------------------------------------------------------------------
# Numeric patterns
# ---------------------------------------------------------------------------

NUM = r"(-?\d+(?:\.\d+)?)"

# (field, pattern, transform). Patterns are matched case-insensitively against
# the whole message; the first capture group is the value.
NUMERIC_PATTERNS: list[tuple[str, str, Any]] = [
    # soil organic carbon: accept SOC, "organic carbon", and organic matter
    # (converted with the conventional 1.724 Van Bemmelen factor)
    ("soil_organic_carbon_pct", rf"\bsoc\b[^\d%]{{0,20}}{NUM}\s*%?", float),
    ("soil_organic_carbon_pct", rf"soil\s+organic\s+carbon[^\d%]{{0,20}}{NUM}\s*%?", float),
    ("soil_organic_carbon_pct", rf"organic\s+carbon[^\d%]{{0,20}}{NUM}\s*%?", float),
    ("soil_organic_carbon_pct", rf"{NUM}\s*%\s*(?:soc|soil\s+organic\s+carbon|organic\s+carbon)", float),
    ("soil_organic_carbon_pct", rf"organic\s+matter[^\d%]{{0,20}}{NUM}\s*%?", lambda v: float(v) / 1.724),
    ("soil_ph", rf"\bph\b[^\d]{{0,15}}{NUM}", float),
    ("soil_ph", rf"{NUM}\s*ph\b", float),
    ("soil_moisture_pct", rf"(?:soil\s+)?moisture[^\d%]{{0,20}}{NUM}\s*%", float),
    ("soil_bulk_density_g_cm3", rf"bulk\s+density[^\d]{{0,20}}{NUM}", float),
    ("soil_nitrogen_pct", rf"(?:soil\s+)?(?:total\s+)?nitrogen[^\d%]{{0,15}}{NUM}\s*%", float),
    ("salinity_ec_ds_m", rf"(?:salinity|\bec\b)[^\d]{{0,20}}{NUM}\s*(?:ds\s*/?\s*m)?", float),
    ("slope_pct", rf"slope[^\d]{{0,20}}{NUM}\s*%?", float),
    ("annual_rainfall_mm", rf"rainfall[^\d]{{0,25}}{NUM}\s*mm", float),
    ("annual_rainfall_mm", rf"{NUM}\s*mm(?:\s*(?:/|per\s*)?(?:yr|year|annum|a))?\s*(?:of\s*)?(?:rain|rainfall|precipitation)", float),
    ("annual_rainfall_mm", rf"(?:rain|precipitation)[^\d]{{0,25}}{NUM}\s*mm", float),
    ("mean_annual_temp_c", rf"(?:mean\s+annual\s+)?temperature[^\d]{{0,20}}{NUM}\s*(?:deg\s*)?c?", float),
    ("dry_season_months", rf"{NUM}\s*(?:dry[- ]season\s*)?months?\s*(?:of\s*)?dry", int),
    ("dry_season_months", rf"dry\s+season[^\d]{{0,20}}{NUM}\s*months?", int),
    ("water_table_depth_m", rf"water\s+table[^\d]{{0,25}}{NUM}\s*m\b", float),
    ("area_ha", rf"{NUM}\s*(?:ha|hectares?)\b", float),
    ("tree_cover_pct", rf"tree\s+(?:cover|canopy)[^\d%]{{0,20}}{NUM}\s*%?", float),
    ("tree_cover_pct", rf"canopy\s+cover[^\d%]{{0,20}}{NUM}\s*%?", float),
    ("native_vegetation_pct", rf"(?:semi[- ]natural|native)\s+(?:vegetation|cover|habitat)[^\d%]{{0,25}}{NUM}\s*%?", float),
    ("mean_patch_size_ha", rf"(?:field|parcel|patch)\s+size[^\d]{{0,20}}{NUM}\s*(?:ha|hectares?)?", float),
    ("hedgerow_density_m_per_ha", rf"hedgerow[^\d]{{0,25}}{NUM}\s*m", float),
    ("distance_to_natural_habitat_m", rf"(?:{NUM}\s*km)\s*(?:away\s*)?(?:from|to)?\s*(?:the\s*)?(?:nearest\s*)?(?:natural|semi[- ]natural|forest|habitat|woodland|wood|scrub|bush|thicket|remnant|uncultivated|wilderness|jungle|grove|copse|shola|savann\w*|native\s+vegetation|wild\s+(?:land|area|patch))", lambda v: float(v) * 1000),
    ("distance_to_natural_habitat_m", rf"(?:natural|semi[- ]natural|forest|habitat|woodland|wood|scrub|bush|thicket|remnant|uncultivated|wilderness|jungle|grove|copse|shola|savann\w*|native\s+vegetation|wild\s+(?:land|area|patch))[^\d]{{0,30}}{NUM}\s*km", lambda v: float(v) * 1000),
    ("distance_to_natural_habitat_m", rf"(?:natural|semi[- ]natural|forest|habitat|woodland|wood|scrub|bush|thicket|remnant|uncultivated|wilderness|jungle|grove|copse|shola|savann\w*|native\s+vegetation|wild\s+(?:land|area|patch))[^\d]{{0,30}}{NUM}\s*m\b", float),
    ("species_richness_observed", rf"{NUM}\s*species", int),
    ("species_richness_observed", rf"species\s+(?:richness|count)[^\d]{{0,20}}{NUM}", int),
    ("bird_species_count", rf"{NUM}\s*bird\s+species", int),
    ("earthworm_count_per_m2", rf"{NUM}\s*earthworms?", float),
    ("earthworm_count_per_m2", rf"earthworms?[^\d]{{0,25}}{NUM}", float),
    ("fertilizer_n_kg_ha_yr", rf"{NUM}\s*kg\s*(?:of\s*)?n\b", float),
    ("fertilizer_n_kg_ha_yr", rf"(?:nitrogen|urea|fertili[sz]er)[^\d]{{0,25}}{NUM}\s*kg", float),
    ("pesticide_applications_per_season", rf"{NUM}\s*(?:pesticide|insecticide|spray|sprays|applications?)", int),
    ("pesticide_applications_per_season", rf"(?:spray|pesticide|insecticide)[^\d]{{0,25}}{NUM}\s*(?:times|applications?|rounds?)", int),
    ("grazing_intensity_lsu_ha", rf"{NUM}\s*(?:lsu|livestock\s+units?|au)\s*(?:/|per\s*)?(?:ha|hectare)?", float),
    ("grazing_intensity_lsu_ha", rf"stocking[^\d]{{0,25}}{NUM}", float),
    ("deforestation_last_5yr_pct", rf"(?:cleared|deforest\w*)[^\d%]{{0,25}}{NUM}\s*%", float),
    ("deforestation_last_5yr_pct", rf"{NUM}\s*%[^.]{{0,25}}(?:cleared|deforest)", float),
]

# ---------------------------------------------------------------------------
# Keyword patterns
# ---------------------------------------------------------------------------

LAND_USE_KEYWORDS: list[tuple[str, LandUse]] = [
    (r"\bmangrove", LandUse.MANGROVE),
    (r"\bwetland|\bmarsh|\bswamp", LandUse.WETLAND),
    (r"\borchard|\bvineyard|\bplantation\s+of\s+fruit", LandUse.ORCHARD),
    (r"\bplantation|\bteak|\beucalypt", LandUse.PLANTATION),
    (r"\bagroforest|\bparkland|\balley\s+crop|\bsilvopast", LandUse.AGROFORESTRY),
    (r"\bforest|\bwoodland", LandUse.FOREST),
    (r"\bpasture|\branch|\bgrazing\s+land", LandUse.PASTURE),
    (r"\bgrassland|\bsavann|\brangeland|\bmeadow", LandUse.GRASSLAND),
    (r"\bbarren|\bwasteland|\bdegraded\s+land|\bfallow\s+degraded|\bbare\s+land", LandUse.DEGRADED),
    (r"\bperi[- ]urban|\burban\s+fringe", LandUse.PERI_URBAN),
    (r"\bcrop|\bfarm|\bfield|\barable|\bwheat|\bmaize|\brice|\bpaddy|\bmillet|\bsorghum|\bcotton|\bsoy", LandUse.CROPLAND),
]

CROPPING_KEYWORDS: list[tuple[str, CroppingSystem]] = [
    (r"\bagroforest|\balley\s+crop|\bparkland|\btrees?\s+(?:on|in)\s+(?:the\s+)?field", CroppingSystem.AGROFORESTRY),
    (r"\bintercrop|\bmixed\s+crop|\bpolycultur|\bcompanion\s+crop", CroppingSystem.INTERCROP),
    (r"\brotat", CroppingSystem.ROTATION),
    (r"\bmonocultur|\bsingle\s+crop|\bcontinuous\s+(?:wheat|maize|rice|cotton|crop)|\bsole\s+crop", CroppingSystem.MONOCULTURE),
    (r"\bfallow", CroppingSystem.FALLOW),
]

CLIMATE_KEYWORDS: list[tuple[str, ClimateZone]] = [
    (r"\bsemi[- ]?arid", ClimateZone.SEMI_ARID),
    (r"\bhyper[- ]?arid|\bdesert|\barid\b", ClimateZone.ARID),
    (r"\bdry\s+sub[- ]?humid", ClimateZone.DRY_SUBHUMID),
    (r"\bsub[- ]?humid", ClimateZone.SUBHUMID),
    (r"\bhumid|\btropical\s+wet|\brainforest", ClimateZone.HUMID),
    (r"\bmontane|\bhighland|\bmountain", ClimateZone.MONTANE),
]

RAINFALL_PATTERN_KEYWORDS: list[tuple[str, RainfallPattern]] = [
    (r"\bbimodal|\btwo\s+rain|\bdouble\s+season", RainfallPattern.BIMODAL),
    (r"\berratic|\bunpredictable\s+rain|\bunreliable\s+rain|\bvariable\s+rain", RainfallPattern.ERRATIC),
    (r"\bunimodal|\bsingle\s+rain|\bone\s+rainy\s+season", RainfallPattern.UNIMODAL),
    (r"\byear[- ]round\s+rain|\baseasonal", RainfallPattern.ASEASONAL),
]

# Qualitative phrasings mapped to representative values. Each is recorded as
# `inferred` provenance so the engine knows it was not a measurement.
QUALITATIVE: list[tuple[str, str, Any]] = [
    ("annual_rainfall_mm", r"\b(?:very\s+low|scarce|minimal)\s+rain", 250.0),
    ("annual_rainfall_mm", r"\blow\s+rain|\brainfall\s+is\s+low|\bpoor\s+rain|\bdry\s+region", 420.0),
    ("annual_rainfall_mm", r"\bhigh\s+rain|\bheavy\s+rain|\babundant\s+rain|\bwell[- ]watered", 1600.0),
    ("pollinator_activity", r"\bno\s+(?:bees|pollinators)|\bpollinators?\s+(?:are\s+)?(?:absent|gone|disappeared)", "absent"),
    ("pollinator_activity", r"\b(?:few|hardly\s+any|very\s+little)\s+(?:bees|pollinator)|\bpollinator\s+activity\s+is\s+low|\blow\s+pollinator", "low"),
    ("pollinator_activity", r"\b(?:plenty|lots|many|abundant)\s+of\s+(?:bees|pollinator)|\bhigh\s+pollinator", "high"),
    ("pollinator_activity", r"\bmoderate\s+pollinator|\bsome\s+bees", "moderate"),
    ("erosion_class", r"\b(?:severe|bad|heavy)\s+erosion|\bgull(?:y|ies)|\bdeep\s+rills", "severe"),
    ("erosion_class", r"\bmoderate\s+erosion|\brills\b|\bsheet\s+erosion", "moderate"),
    ("erosion_class", r"\bslight\s+erosion|\bminor\s+erosion|\ba\s+little\s+erosion", "slight"),
    ("erosion_class", r"\bno\s+erosion", "none"),
    ("tillage", r"\bno[- ]till|\bzero\s+till|\bdirect\s+(?:seed|drill)", "none"),
    ("tillage", r"\bminimum\s+till|\breduced\s+till|\bstrip\s+till|\bshallow\s+till", "reduced"),
    ("tillage", r"\bdeep\s+(?:till|plough|plow)|\bsubsoil|\bmould\s*board", "deep"),
    ("tillage", r"\bplough|\bplow|\bconventional\s+till|\btilled?\b|\bdisc(?:ing)?\b", "conventional"),
    ("residue_removal", r"\bburn(?:ing|ed|t)?\s+(?:the\s+)?(?:residue|stubble|straw)|\bstubble\s+burn", "burned"),
    ("residue_removal", r"\bresidues?\s+(?:are\s+)?(?:fully\s+)?removed|\bremove\s+(?:all\s+)?(?:the\s+)?(?:residue|straw|stover)|\bsold?\s+(?:the\s+)?(?:straw|stover)|\bfodder\s+for\s+(?:cattle|livestock)", "full"),
    ("residue_removal", r"\bsome\s+residue|\bpartly\s+removed|\bpartially\s+removed", "partial"),
    ("residue_removal", r"\bresidues?\s+(?:are\s+)?(?:left|retained)|\bmulch(?:ed|ing)?\b|\bleave\s+(?:the\s+)?(?:residue|straw)", "none"),
    ("irrigation", r"\bdrip\s+irrigat", "drip"),
    ("irrigation", r"\bsprinkler|\bpivot\s+irrigat", "sprinkler"),
    ("irrigation", r"\bflood\s+irrigat|\bbasin\s+irrigat", "flood"),
    ("irrigation", r"\bfurrow\s+irrigat", "furrow"),
    ("irrigation", r"\bsupplement(?:al|ary)\s+irrigat", "rainfed_supplemental"),
    ("irrigation", r"\brainfed|\brain[- ]fed|\bno\s+irrigation|\bunirrigated", "none"),
    ("soil_texture", r"\bsandy\s+loam", "sandy_loam"),
    ("soil_texture", r"\bsilt(?:y)?\s+loam", "silt_loam"),
    ("soil_texture", r"\bclay\s+loam", "clay_loam"),
    ("soil_texture", r"\bsandy\s+soil|\bsand\b", "sand"),
    ("soil_texture", r"\bclay\s+soil|\bheavy\s+clay|\bblack\s+cotton\s+soil|\bvertisol", "clay"),
    ("soil_texture", r"\bloam(?:y)?\b", "loam"),
]

# Negations are listed FIRST for each field: "no invasive weeds" contains the
# token "invasive", so a positive pattern evaluated first would invert the
# meaning. First match wins per field, so order encodes precedence.
BOOLEAN_PATTERNS: list[tuple[str, str, bool]] = [
    ("invasive_species_present", r"\b(?:no|not\s+any|without)\s+(?:aggressive\s+|exotic\s+)?invasiv|\bno\s+(?:invasive\s+)?weeds?\b", False),
    ("invasive_species_present", r"\binvasiv|\blantana|\bprosopis|\bparthenium|\bwater\s+hyacinth|\bexotic\s+weed|\bweeds?\s+taking\s+over", True),
    ("permanent_water_present", r"\bno\s+(?:permanent\s+|seasonal\s+|surface\s+)?water|\bno\s+pond|\bno\s+stream|\bwater\s+source\s+(?:is\s+)?absent|\bdry\s+all\s+year", False),
    ("permanent_water_present", r"\b(?:pond|stream|river|lake|tank|spring|canal|water\s+body|wetland)\b", True),
]

CROP_KEYWORDS = (
    "wheat", "maize", "corn", "rice", "paddy", "millet", "sorghum", "barley", "cotton",
    "soybean", "soya", "groundnut", "peanut", "chickpea", "pigeonpea", "lentil", "mustard",
    "sugarcane", "coffee", "tea", "cocoa", "oil palm", "banana", "cassava", "potato",
    "tomato", "onion", "apple", "mango", "citrus", "grape", "almond", "olive", "sunflower",
    "canola", "oilseed rape", "teff", "sesame", "cowpea",
)

_GEO_RE = re.compile(
    r"(-?\d{1,2}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)"
)
_GEO_DMS_RE = re.compile(
    r"(\d{1,2}(?:\.\d+)?)\s*[^\w\s]?\s*([NS])[,\s]+(\d{1,3}(?:\.\d+)?)\s*[^\w\s]?\s*([EW])",
    re.IGNORECASE,
)

OBJECTIVE_KEYWORDS: list[tuple[str, str]] = [
    (r"\bcarbon\s+credit|\bcarbon\s+finance|\bcarbon\s+market", "carbon credits / monetisation"),
    (r"\bsequester|\bsoil\s+carbon\s+goal|\bcarbon\s+stock", "soil carbon sequestration"),
    (r"\bbiodiversity|\bspecies|\bwildlife|\bhabitat", "biodiversity recovery"),
    (r"\byield|\bproductivity|\bproduction|\bincome|\bprofit", "maintain or raise production"),
    (r"\bwater|\bdrought|\birrigation\s+cost", "water security"),
    (r"\berosion|\bsoil\s+loss|\btopsoil", "erosion control"),
    (r"\bpollinator|\bbees?\b", "pollination services"),
    (r"\bcertif|\bcompliance|\bregulat|\bESG|\breporting", "certification / reporting"),
]

CONSTRAINT_KEYWORDS: list[tuple[str, str]] = [
    (r"\blimited\s+labour|\bno\s+labour|\blabour\s+short|\blabor\s+short", "labour-constrained"),
    (r"\blow\s+budget|\bno\s+money|\bcannot\s+afford|\blimited\s+(?:budget|capital|funds)|\bcheap", "capital-constrained"),
    (r"\bsmallholder|\bsmall\s+farm|\bhalf\s+a\s+hectare|\bone\s+hectare", "smallholder scale"),
    (r"\bno\s+machinery|\bhand\s+tools|\bmanual", "no machinery access"),
    (r"\brent|\bleased?\s+land|\btenure|\bno\s+title", "insecure tenure"),
    (r"\blivestock\s+need|\bfodder\s+need|\bneed\s+the\s+(?:straw|stover|residue)", "residues needed as fodder"),
]


def _first_match(text: str, patterns: list[tuple[str, Any]]) -> Any | None:
    for pattern, value in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def rule_extract(message: str) -> SiteProfile:
    """Deterministic extraction. Never raises on bad input; unparsable values
    are simply not extracted."""
    text = message.lower()
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    # --- numeric, first match wins per field ---
    for field, pattern, transform in NUMERIC_PATTERNS:
        if field in values:
            continue
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            values[field] = transform(match.group(1))
            provenance[field] = "user"
        except (TypeError, ValueError):
            continue

    # --- categorical keywords ---
    for field, patterns in (
        ("land_use", LAND_USE_KEYWORDS),
        ("cropping_system", CROPPING_KEYWORDS),
        ("climate_zone", CLIMATE_KEYWORDS),
        ("rainfall_pattern", RAINFALL_PATTERN_KEYWORDS),
    ):
        if field in values:
            continue
        found = _first_match(text, list(patterns))
        if found is not None:
            values[field] = found
            provenance[field] = "user"

    # --- qualitative phrasings (weaker: recorded as inferred) ---
    for field, pattern, value in QUALITATIVE:
        if field in values:
            continue
        if re.search(pattern, text, re.IGNORECASE):
            values[field] = value
            provenance[field] = "inferred" if isinstance(value, float) else "user"

    # --- booleans ---
    for field, pattern, value in BOOLEAN_PATTERNS:
        if field in values:
            continue
        if re.search(pattern, text, re.IGNORECASE):
            values[field] = value
            provenance[field] = "user"

    # --- primary crop ---
    for crop in CROP_KEYWORDS:
        if re.search(rf"\b{re.escape(crop)}\b", text):
            values["primary_crop"] = crop
            provenance["primary_crop"] = "user"
            break

    # --- objectives and constraints (accumulate, never overwrite) ---
    objectives = [label for pattern, label in OBJECTIVE_KEYWORDS if re.search(pattern, text, re.IGNORECASE)]
    constraints = [label for pattern, label in CONSTRAINT_KEYWORDS if re.search(pattern, text, re.IGNORECASE)]

    # --- geo coordinates ---
    geo: dict[str, Any] = {}
    dms = _GEO_DMS_RE.search(message)
    if dms:
        lat = float(dms.group(1)) * (-1 if dms.group(2).upper() == "S" else 1)
        lon = float(dms.group(3)) * (-1 if dms.group(4).upper() == "W" else 1)
        geo = {"latitude": lat, "longitude": lon}
    else:
        # Only treat a bare number pair as coordinates when it is explicitly
        # labelled, otherwise "5, 20" in prose would be misread as a location.
        if re.search(r"\b(?:lat|latitude|coordinates?|geo|gps|located\s+at)\b", text):
            plain = _GEO_RE.search(message)
            if plain:
                lat, lon = float(plain.group(1)), float(plain.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    geo = {"latitude": lat, "longitude": lon}

    # Sanity-clamp values that the schema would reject, rather than dropping the
    # whole extraction because of one implausible number.
    values = _sanitise(values)

    payload: dict[str, Any] = {**values, "provenance": provenance}
    if objectives:
        payload["objectives"] = objectives
    if constraints:
        payload["constraints"] = constraints
    if geo:
        payload["geo"] = geo

    try:
        return SiteProfile(**payload)
    except Exception:
        # Fall back to a field-by-field build so a single bad value cannot void
        # everything the user said.
        safe: dict[str, Any] = {"provenance": provenance}
        for key, value in payload.items():
            if key == "provenance":
                continue
            try:
                SiteProfile(**{key: value})
                safe[key] = value
            except Exception:
                continue
        return SiteProfile(**safe)


_BOUNDS: dict[str, tuple[float, float]] = {
    "soil_organic_carbon_pct": (0, 60),
    "soil_ph": (2, 11),
    "soil_moisture_pct": (0, 100),
    "soil_bulk_density_g_cm3": (0.5, 2.2),
    "soil_nitrogen_pct": (0, 5),
    "salinity_ec_ds_m": (0, 60),
    "slope_pct": (0, 100),
    "annual_rainfall_mm": (0, 12000),
    "mean_annual_temp_c": (-20, 45),
    "dry_season_months": (0, 12),
    "water_table_depth_m": (0, 500),
    "tree_cover_pct": (0, 100),
    "native_vegetation_pct": (0, 100),
    "fertilizer_n_kg_ha_yr": (0, 1000),
    "pesticide_applications_per_season": (0, 40),
    "grazing_intensity_lsu_ha": (0, 10),
    "deforestation_last_5yr_pct": (0, 100),
}


def _sanitise(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        bounds = _BOUNDS.get(key)
        if bounds and isinstance(value, (int, float)):
            low, high = bounds
            if not (low <= value <= high):
                continue  # implausible; drop rather than clamp silently
        if key in ("area_ha", "mean_patch_size_ha") and isinstance(value, (int, float)) and value <= 0:
            continue
        out[key] = value
    return out


def extract(message: str, use_llm: bool = True) -> tuple[SiteProfile, bool]:
    """Extract a profile patch from one user message.

    Returns (patch, llm_used). Rule extraction always runs; the LLM only adds
    fields the rules did not find.
    """
    patch = rule_extract(message)
    if not use_llm:
        return patch, False

    from ..llm.client import get_client

    client = get_client()
    if not client.available:
        return patch, False

    llm_patch = client.extract_profile(message, already_found=set(patch.known()))
    if llm_patch is None:
        return patch, False
    # Rule-extracted values win: merge the LLM patch in *under* them.
    return llm_patch.merge(patch), True
