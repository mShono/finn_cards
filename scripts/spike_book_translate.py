"""P0 spike: translate a few chapters of a book and write a reading report.

Not part of the bot - a throwaway driver for kielikaveri.translate to judge
translation quality and cost before any UI exists (plan
2026-11-20-kirjakaveri-reader.md, "П0"). Output goes next to the book, never
into the repo: the repo is public and the book is copyrighted.

Usage:
    uv run python scripts/spike_book_translate.py BOOK.txt --chapters 1,34 \
        [--no-fst] [--no-review] [--edit] [--price-in 1.25 --price-out 10]

--chapters are running numbers across the book (see `--list`). Prices are
USD per 1M tokens; without them only token counts are reported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from kielikaveri.config import Settings
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.llm.client import make_client
from kielikaveri.translate.chapters import read_text, split_book
from kielikaveri.translate.pipeline import (
    RefusedError,
    Usage,
    prepare_guide,
    translate_chapter,
)

# A chapter of ~25k chars is ~5 blocks x (translate + review) plus repairs;
# the bot's own 60/10min budget would trip on the first chapter.
SPIKE_MAX_CALLS = 300


def _report(result) -> str:
    chapter = result.chapter
    lines = [f"# Глава {chapter.number} ({chapter.part or '-'}, «{chapter.title}»)", ""]
    lines.append(
        f"Абзацев: {len(chapter.paragraphs)}. FST-флаги в черновике: {len(result.fst_flags)} абз., "
        f"после всех шагов: {len(result.fst_flags_final)} абз. Замечаний ревью: {len(result.issues)}."
    )
    lines.append("")
    issues_by_id: dict[int, list] = {}
    for issue in result.issues:
        issues_by_id.setdefault(issue.paragraph_id, []).append(issue)
    for pid, source in enumerate(chapter.paragraphs):
        lines.append(f"## ¶{pid}")
        lines.append(f"**RU:** {source}")
        lines.append("")
        draft, final = result.draft.get(pid, ""), result.final.get(pid, "")
        if draft != final:
            lines.append(f"**FI черновик:** {draft}")
            lines.append("")
        lines.append(f"**FI:** {final}")
        if pid in result.fst_flags:
            lines.append(f"- FST не узнал в черновике: {', '.join(result.fst_flags[pid])}")
        if pid in result.fst_sent:
            lines.append(f"- отправлено на исправление: {', '.join(result.fst_sent[pid])}")
        if pid in result.fst_flags_final:
            lines.append(f"- FST не узнал в итоге: {', '.join(result.fst_flags_final[pid])}")
        for issue in issues_by_id.get(pid, []):
            lines.append(f"- ревью [{issue.category}] «{issue.quote_fi}»: {issue.explanation_ru}")
        lines.append("")
    return "\n".join(lines)


def _usage_table(usage: Usage, price_in: float | None, price_out: float | None) -> str:
    rows = ["| шаг | вызовов | вход, ток. | выход, ток. | сек | $ |", "|---|---|---|---|---|---|"]
    total = 0.0
    for name, step in usage.steps.items():
        cost = ""
        if price_in is not None and price_out is not None:
            dollars = (step.input_tokens * price_in + step.output_tokens * price_out) / 1_000_000
            total += dollars
            cost = f"{dollars:.3f}"
        rows.append(
            f"| {name} | {step.calls} | {step.input_tokens} | {step.output_tokens} "
            f"| {step.seconds:.0f} | {cost} |"
        )
    if price_in is not None:
        rows.append(f"| **итого** | | | | | **{total:.3f}** |")
    return "\n".join(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("book", type=Path)
    parser.add_argument("--list", action="store_true", help="only print the chapter list")
    parser.add_argument("--chapters", default="1")
    parser.add_argument("--no-fst", action="store_true")
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--price-in", type=float, default=None)
    parser.add_argument("--price-out", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    book = split_book(read_text(args.book))
    if args.list:
        for chapter in book.chapters:
            print(
                f"{chapter.number:3d}  {chapter.part or '-':28s} {chapter.title:>4s}  {chapter.chars:6d} зн."
            )
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Library loggers stay quiet: openai's DEBUG would dump every prompt (plan, "Логи").
    for name in ("openai", "httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)

    settings = Settings()
    model = args.model or settings.openai_text_model
    client = make_client(settings.openai_api_key, timeout_seconds=300)
    breaker = CallBreaker(SPIKE_MAX_CALLS, timedelta(minutes=10))
    wanted = {int(n) for n in args.chapters.split(",")}
    chapters = [c for c in book.chapters if c.number in wanted]
    out = args.out or args.book.parent / "spike" / args.book.stem
    out.mkdir(parents=True, exist_ok=True)

    usage = Usage()
    guide = await prepare_guide(client, breaker, model, usage, book, chapters)
    (out / "guide.json").write_text(
        json.dumps(asdict(guide), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = [f"# Спайк перевода: {args.book.stem}", "", f"Модель: `{model}`", ""]
    for chapter in chapters:
        try:
            result = await translate_chapter(
                client,
                breaker,
                model,
                usage,
                guide,
                chapter,
                fst=not args.no_fst,
                review=not args.no_review,
                edit=args.edit,
            )
        except RefusedError as exc:
            summary.append(f"- Глава {chapter.number}: **модель отказалась** ({exc})")
            continue
        (out / f"chapter_{chapter.number:02d}.md").write_text(_report(result), encoding="utf-8")
        summary.append(
            f"- Глава {chapter.number}: {len(chapter.paragraphs)} абз., FST-флаги "
            f"{len(result.fst_flags)} → {len(result.fst_flags_final)}, ревью {len(result.issues)}"
        )
    summary += ["", _usage_table(usage, args.price_in, args.price_out), ""]
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"готово: {out}")


if __name__ == "__main__":
    asyncio.run(main())
