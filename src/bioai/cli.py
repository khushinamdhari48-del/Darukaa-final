"""Terminal client.

    python -m bioai.cli chat                      interactive multi-turn session
    python -m bioai.cli assess --json site.json   one-shot structured assessment
    python -m bioai.cli demo [1-5]                run a scripted scenario
    python -m bioai.cli search "query"            inspect the retrieval layer
    python -m bioai.cli stats                     knowledge base composition
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .dialogue.orchestrator import Orchestrator
from .dialogue.render import render_assessment
from .knowledge.loader import load_knowledge_base
from .knowledge.retriever import get_index
from .llm.client import get_client
from .reasoning.engine import get_engine
from .schemas import ChatRequest, SiteProfile

console = Console()

BANNER = """[bold green]Darukaa Biodiversity Intelligence Engine[/bold green]
Evidence-grounded environmental advisory. Recommendations come from a curated,
citation-carrying knowledge base and a deterministic reasoning engine.

Commands: [cyan]/profile[/cyan] show accumulated site memory | [cyan]/why[/cyan] show scoring traces
          [cyan]/reset[/cyan] start over | [cyan]/quit[/cyan] exit
"""


def _status_line() -> str:
    idx = get_index()
    kb = load_knowledge_base()
    return (
        f"{kb.stats()['evidence_cards']} evidence cards | "
        f"{kb.stats()['causal_edges']} causal edges | "
        f"{kb.stats()['interventions']} interventions | "
        f"embedder={idx.embedder.name} | vector store={idx.store.backend} | "
        f"LLM: {get_client().status}"
    )


def cmd_chat(args: argparse.Namespace) -> int:
    orchestrator = Orchestrator()
    console.print(Panel(BANNER, border_style="green"))
    console.print(f"[dim]{_status_line()}[/dim]\n")

    session_id: str | None = None
    explain = False
    while True:
        try:
            message = console.input("[bold cyan]you >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]session ended[/dim]")
            return 0
        if not message:
            continue
        lowered = message.lower()
        if lowered in ("/quit", "/exit", "quit", "exit"):
            return 0
        if lowered == "/reset":
            session_id = None
            console.print("[yellow]session reset[/yellow]\n")
            continue
        if lowered == "/why":
            explain = not explain
            console.print(f"[yellow]scoring traces {'on' if explain else 'off'}[/yellow]\n")
            continue
        if lowered == "/profile":
            if session_id is None:
                console.print("[dim]nothing captured yet[/dim]\n")
                continue
            session = orchestrator.store.get(session_id)
            console.print(
                Panel(
                    json.dumps(session.profile.known(), indent=2, default=str),
                    title="accumulated site memory",
                    border_style="blue",
                )
            )
            continue

        with console.status("[dim]retrieving evidence and reasoning...[/dim]"):
            response = orchestrator.handle(
                ChatRequest(session_id=session_id, message=message, explain_retrieval=explain)
            )
        session_id = response.session_id
        console.print()
        console.print(Markdown(response.reply))
        console.print(
            f"[dim]turn {response.turn} | session {session_id} | "
            f"llm={'yes' if response.llm_used else 'no (deterministic)'}[/dim]\n"
        )


def cmd_assess(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.json).read_text(encoding="utf-8")) if args.json else json.load(sys.stdin)
    profile = SiteProfile(**raw)
    assessment = get_engine().assess(
        profile=profile,
        query_hint=args.query or "",
        max_recommendations=args.max_recommendations,
        explain_retrieval=args.explain,
    )
    if args.format == "json":
        console.print_json(assessment.model_dump_json())
    else:
        console.print(Markdown(render_assessment(assessment, include_trace=args.explain)))
    return 0


DEMOS: dict[str, tuple[str, list[str]]] = {
    "1": (
        "Semi-arid monoculture wheat with depleted soil carbon (the brief's worked example)",
        [
            "Biodiversity is declining on my land.",
            "Soil organic carbon is 0.3%, rainfall is low at about 420 mm, I grow monoculture "
            "wheat, semi-arid region, pH 5.2.",
            "The nearest patch of scrub is about 1.5 km away, I spray 4 times a season, "
            "and I burn the stubble. No irrigation.",
        ],
    ),
    "2": (
        "Over-fertilised temperate cropland - the opposite prescription to demo 1",
        [
            "I farm 300 hectares of cereals in a temperate region, rainfall 850 mm, "
            "soil organic carbon 2.4%, pH 6.8. I apply 220 kg N per hectare and spray 5 times "
            "a season. Mean field size is 40 ha and there are no hedgerows. Pollinators are low.",
        ],
    ),
    "3": (
        "Overgrazed semi-arid rangeland",
        [
            "I have semi-arid grassland, 480 mm rainfall, stocking at 2.2 LSU/ha. "
            "Soil organic carbon 0.9%, bulk density 1.55, severe erosion with gullies. "
            "No surface water. Invasive weeds are taking over.",
        ],
    ),
    "4": (
        "Salt-affected irrigated land with a shallow water table",
        [
            "Irrigated cotton, flood irrigation, salinity EC 9 dS/m, pH 8.4, "
            "water table 1.2 m, rainfall 300 mm, soil organic carbon 0.6%.",
        ],
    ),
    "5": (
        "Geo-referenced humid smallholder plot with structured follow-up",
        [
            "Located at 0.35 N, 37.58 E. Two hectares, humid, rainfall 1700 mm, "
            "maize monoculture, pH 5.1, organic matter 2.8%, no irrigation, slope 18%. "
            "Limited budget and no machinery. I want biodiversity and yield.",
        ],
    ),
}


def cmd_demo(args: argparse.Namespace) -> int:
    key = args.scenario
    if key not in DEMOS:
        console.print("[red]unknown scenario[/red]. Available:")
        for k, (title, _) in DEMOS.items():
            console.print(f"  [cyan]{k}[/cyan]  {title}")
        return 1
    title, messages = DEMOS[key]
    console.print(Panel(f"[bold]{title}[/bold]\n\n[dim]{_status_line()}[/dim]", border_style="green"))
    orchestrator = Orchestrator()
    session_id: str | None = None
    for message in messages:
        console.print(Panel(message, title="user", border_style="cyan"))
        response = orchestrator.handle(
            ChatRequest(session_id=session_id, message=message, explain_retrieval=args.explain)
        )
        session_id = response.session_id
        console.print(Markdown(response.reply))
        console.print(
            f"[dim]turn {response.turn} | llm={'yes' if response.llm_used else 'no (deterministic)'}[/dim]\n"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    idx = get_index()
    profile = None
    if args.profile:
        profile = SiteProfile(**json.loads(Path(args.profile).read_text(encoding="utf-8")))
    results, trace = idx.retrieve(
        query=args.query, profile=profile, top_k=args.top_k, include_violated=args.include_violated
    )
    table = Table(title=f"retrieval: {args.query!r}", show_lines=False)
    table.add_column("fused", justify="right")
    table.add_column("dense", justify="right")
    table.add_column("lex", justify="right")
    table.add_column("cond", justify="right")
    table.add_column("path")
    table.add_column("citation")
    table.add_column("claim", overflow="fold", max_width=60)
    for r in results:
        table.add_row(
            f"{r.score:.3f}",
            f"{r.dense_score:.2f}",
            f"{r.lexical_score:.2f}",
            f"{r.condition_score:.2f}",
            r.retrieval_path,
            r.card.citation.short(),
            r.card.claim.strip()[:220],
        )
    console.print(table)
    console.print(
        f"[dim]weights {trace.weights} | embedder={trace.embedder} | "
        f"store={trace.vector_backend} | corpus={trace.n_candidates} cards[/dim]"
    )
    if profile is not None:
        contradicted = idx.find_contradicted(profile)
        if contradicted:
            console.print("\n[yellow]Evidence explicitly inapplicable to this site:[/yellow]")
            for r in contradicted[:8]:
                console.print(f"  [dim]{r.card.id}[/dim] -> {'; '.join(r.violated_conditions)}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    kb = load_knowledge_base()
    stats = kb.stats()
    table = Table(title="knowledge base")
    table.add_column("component")
    table.add_column("count", justify="right")
    for key, value in stats.items():
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)

    by_type: dict[str, int] = {}
    for card in kb.cards:
        by_type[card.citation.type.value] = by_type.get(card.citation.type.value, 0) + 1
    type_table = Table(title="evidence cards by inferential strength")
    type_table.add_column("evidence type")
    type_table.add_column("cards", justify="right")
    for key, value in sorted(by_type.items(), key=lambda kv: -kv[1]):
        type_table.add_row(key.replace("_", " "), str(value))
    console.print(type_table)
    console.print(f"[dim]{_status_line()}[/dim]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="interactive multi-turn session")
    p_chat.set_defaults(func=cmd_chat)

    p_assess = sub.add_parser("assess", help="one-shot structured assessment")
    p_assess.add_argument("--json", help="path to a site profile JSON file (default: stdin)")
    p_assess.add_argument("--query", help="free-text context to steer retrieval")
    p_assess.add_argument("--max-recommendations", type=int, default=4)
    p_assess.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_assess.add_argument("--explain", action="store_true", help="include scoring traces")
    p_assess.set_defaults(func=cmd_assess)

    p_demo = sub.add_parser("demo", help="run a scripted scenario")
    p_demo.add_argument("scenario", nargs="?", default="1", choices=list(DEMOS))
    p_demo.add_argument("--explain", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_search = sub.add_parser("search", help="inspect the retrieval layer")
    p_search.add_argument("query")
    p_search.add_argument("--profile", help="path to a site profile JSON, to condition retrieval")
    p_search.add_argument("--top-k", type=int, default=8)
    p_search.add_argument("--include-violated", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="knowledge base composition")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
