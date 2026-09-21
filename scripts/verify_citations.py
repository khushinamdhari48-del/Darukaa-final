#!/usr/bin/env python
"""Resolve every DOI and URL in the evidence corpus over the network.

The corpus claims real, published sources. This script lets a reviewer verify
that claim without taking it on trust: it resolves each DOI against
doi.org/api.crossref.org and reports the registered title alongside the title
recorded in the corpus, flagging any mismatch.

    python scripts/verify_citations.py                 # DOIs only (fast)
    python scripts/verify_citations.py --check-urls    # also HEAD each report URL
    python scripts/verify_citations.py --json report.json

Requires network access. It is deliberately NOT part of the test suite or CI,
because tests must pass offline and must not depend on third-party uptime.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from bioai.knowledge.loader import load_knowledge_base

CROSSREF = "https://api.crossref.org/works/{doi}"
# Content negotiation against doi.org resolves every registration agency, not
# just Crossref. Zenodo, Dryad and most institutional DOIs are DataCite and are
# invisible to the Crossref API.
DOI_ORG = "https://doi.org/{doi}"
USER_AGENT = "bioai-citation-verifier/1.0 (biodiversity knowledge base integrity check)"

# Crossref's `issued` date is frequently the online-first date, which is often
# the year before the print issue. A one-year gap with an identical title is an
# indexing artefact, not a wrong citation.
YEAR_TOLERANCE = 1


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", title.lower()).strip()


def _similarity(a: str, b: str) -> float:
    a_norm, b_norm = _normalise(a), _normalise(b)
    if not a_norm or not b_norm:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    # A chapter cited within a named report legitimately carries a longer title
    # than the registered record (or the reverse), so containment counts as a
    # match rather than as a discrepancy.
    if a_norm in b_norm or b_norm in a_norm:
        return max(ratio, 0.95)
    return ratio


def _parse_crossref(message: dict) -> dict:
    titles = message.get("title") or []
    year = None
    issued = message.get("issued", {}).get("date-parts") or [[None]]
    if issued and issued[0]:
        year = issued[0][0]
    return {
        "status": "resolved",
        "source": "crossref",
        "registered_title": titles[0] if titles else None,
        "registered_year": year,
        "container": (message.get("container-title") or [None])[0],
    }


def _parse_citeproc(payload: dict) -> dict:
    title = payload.get("title")
    if isinstance(title, list):
        title = title[0] if title else None
    year = None
    issued = (payload.get("issued") or {}).get("date-parts") or [[None]]
    if issued and issued[0]:
        year = issued[0][0]
    container = payload.get("container-title") or payload.get("publisher")
    if isinstance(container, list):
        container = container[0] if container else None
    return {
        "status": "resolved",
        "source": "doi.org",
        "registered_title": title,
        "registered_year": year,
        "container": container,
    }


def verify_doi(client: httpx.Client, doi: str) -> dict:
    """Try Crossref first (richest metadata), then doi.org content negotiation."""
    try:
        response = client.get(CROSSREF.format(doi=doi), timeout=25.0)
        if response.status_code == 200:
            try:
                return _parse_crossref(response.json()["message"])
            except (ValueError, KeyError):
                pass
    except httpx.HTTPError:
        pass

    try:
        response = client.get(
            DOI_ORG.format(doi=doi),
            timeout=25.0,
            follow_redirects=True,
            headers={"Accept": "application/vnd.citationstyles.csl+json"},
        )
    except httpx.HTTPError as exc:
        return {"status": "network_error", "detail": str(exc)}
    if response.status_code == 404:
        return {"status": "not_found"}
    if response.status_code != 200:
        return {"status": "http_error", "detail": f"HTTP {response.status_code}"}
    try:
        return _parse_citeproc(response.json())
    except ValueError:
        return {"status": "unparsable_response"}


def verify_url(client: httpx.Client, url: str) -> dict:
    try:
        response = client.head(url, timeout=25.0, follow_redirects=True)
        if response.status_code >= 400:
            # Some institutional servers reject HEAD; retry with a ranged GET.
            response = client.get(
                url, timeout=25.0, follow_redirects=True, headers={"Range": "bytes=0-256"}
            )
        return {"status": "ok" if response.status_code < 400 else "http_error",
                "code": response.status_code}
    except httpx.HTTPError as exc:
        return {"status": "network_error", "detail": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-urls", action="store_true", help="also verify report URLs")
    parser.add_argument("--json", help="write the full report to this path")
    parser.add_argument(
        "--title-threshold", type=float, default=0.80,
        help="similarity below which a title is reported as a possible mismatch",
    )
    args = parser.parse_args()

    kb = load_knowledge_base()
    # Deduplicate: several cards legitimately share a source (e.g. the IPBES
    # Global Assessment), and there is no point resolving the same DOI five times.
    by_doi: dict[str, list[str]] = {}
    by_url: dict[str, list[str]] = {}
    for card in kb.cards:
        if card.citation.doi:
            by_doi.setdefault(card.citation.doi, []).append(card.id)
        elif card.citation.url:
            by_url.setdefault(card.citation.url, []).append(card.id)

    report: dict[str, list[dict]] = {"dois": [], "urls": []}
    resolved = mismatched = failed = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        print(f"Resolving {len(by_doi)} distinct DOIs...\n")
        for doi, card_ids in sorted(by_doi.items()):
            citation = kb.cards_by_id[card_ids[0]].citation
            result = verify_doi(client, doi)
            entry = {"doi": doi, "cards": card_ids, "corpus_title": citation.title, **result}

            if result["status"] != "resolved":
                failed += 1
                print(f"  FAIL      {doi}  ({result['status']}) -> {card_ids}")
            else:
                score = _similarity(citation.title, result.get("registered_title") or "")
                entry["title_similarity"] = round(score, 3)
                registered_year = result.get("registered_year")
                year_delta = (
                    abs(registered_year - citation.year) if registered_year else 0
                )
                entry["year_delta"] = year_delta
                if score >= args.title_threshold and year_delta <= YEAR_TOLERANCE:
                    resolved += 1
                    note = (
                        f"  (year differs by {year_delta}: registered {registered_year}, "
                        f"likely online-first vs issue date)"
                        if year_delta
                        else ""
                    )
                    print(f"  OK        {doi}  {citation.short()}{note}")
                else:
                    mismatched += 1
                    print(f"  MISMATCH  {doi}  similarity={score:.2f}")
                    print(f"            corpus:     {citation.title} ({citation.year})")
                    print(
                        f"            registered: {result.get('registered_title')} "
                        f"({registered_year})"
                    )
            report["dois"].append(entry)

        if args.check_urls:
            print(f"\nChecking {len(by_url)} distinct report URLs...\n")
            for url, card_ids in sorted(by_url.items()):
                result = verify_url(client, url)
                report["urls"].append({"url": url, "cards": card_ids, **result})
                label = "OK      " if result["status"] == "ok" else "FAIL    "
                print(f"  {label}  {url}  ({result.get('code', result['status'])})")

    total = len(by_doi)
    print(
        f"\n{resolved}/{total} DOIs resolved with a matching title, "
        f"{mismatched} possible mismatch(es), {failed} unresolvable."
    )
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full report written to {args.json}")

    # A network failure is not a corpus failure, so only real mismatches fail.
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
