"""Wake-word utilities for Sakura voice mode."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from openjarvis.core.config import get_config_dir


@click.group(help="Wake-word tools for Sakura voice mode.")
def wake() -> None:
    """Manage wake-word data and diagnostics."""


@wake.command("record-samples")
@click.option(
    "--kind",
    type=click.Choice(["positive", "negative"], case_sensitive=False),
    default="positive",
    show_default=True,
    help="Record wake phrase samples or non-wake/background samples.",
)
@click.option(
    "--count",
    type=int,
    default=25,
    show_default=True,
    help="Number of clips to record.",
)
@click.option(
    "--phrase",
    default="hey sakura",
    show_default=True,
    help="Wake phrase to show while recording positive samples.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for samples. Defaults to ~/.openjarvis/wake_samples.",
)
def record_samples(
    kind: str,
    count: int,
    phrase: str,
    output_dir: Path | None,
) -> None:
    """Record WAV clips for custom openWakeWord training."""
    from openjarvis.speech.voice_io import record_until_silence, wav_duration_seconds

    console = Console(stderr=True)
    if count <= 0:
        raise click.BadParameter("count must be greater than zero")

    root = output_dir or (get_config_dir() / "wake_samples")
    target = root / kind.lower()
    target.mkdir(parents=True, exist_ok=True)

    if kind.lower() == "positive":
        console.print(
            f"[cyan]Recording {count} wake samples.[/cyan] Say "
            f"[bold]{phrase!r}[/bold] naturally after each prompt."
        )
    else:
        console.print(
            f"[cyan]Recording {count} negative samples.[/cyan] Use silence, "
            "room noise, normal speech, and similar-but-wrong phrases."
        )

    for index in range(1, count + 1):
        input(f"[{index}/{count}] Press Enter, then speak...")
        audio = record_until_silence(
            silence_seconds=0.7,
            startup_silence_seconds=6.0,
            max_seconds=3.5,
        )
        duration = wav_duration_seconds(audio)
        if duration <= 0:
            console.print("[yellow]Nothing heard; skipping this clip.[/yellow]")
            continue

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = target / f"{kind.lower()}-{stamp}.wav"
        path.write_bytes(audio)
        console.print(f"[green]Saved[/green] {path} ({duration:.2f}s)")

    console.print(f"[cyan]Samples saved under:[/cyan] {root}")


@wake.command("analyze-samples")
@click.option(
    "--samples-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing positive/ and optional negative/ WAV samples.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.82,
    show_default=True,
    help="Sample-match threshold to evaluate.",
)
@click.option(
    "--margin",
    type=float,
    default=0.03,
    show_default=True,
    help="Minimum positive-vs-negative margin to evaluate.",
)
def analyze_samples(
    samples_dir: Path | None,
    threshold: float,
    margin: float,
) -> None:
    """Analyze wake samples and print threshold guidance."""
    from openjarvis.cli._voice_chat import _WakeSampleMatcher

    console = Console(stderr=True)
    root = samples_dir or (get_config_dir() / "wake_samples")
    positive_dir = root / "positive"
    negative_dir = root / "negative"
    positives = sorted(positive_dir.glob("*.wav")) if positive_dir.exists() else []
    negatives = sorted(negative_dir.glob("*.wav")) if negative_dir.exists() else []

    console.print(f"[cyan]Samples:[/cyan] {root}")
    console.print(f"  positive: [bold]{len(positives)}[/bold]")
    console.print(f"  negative: [bold]{len(negatives)}[/bold]")
    if len(positives) < 3:
        console.print("[yellow]Record at least 10 positive samples first.[/yellow]")
        return

    matcher = _WakeSampleMatcher(root)
    rows: list[tuple[str, float, float | None, float, bool]] = []
    for path in positives:
        score, negative_score, score_margin = matcher.score_path(path, exclude_self=True)
        matched = score >= threshold and score_margin >= margin
        rows.append((path.name, score, negative_score, score_margin, matched))
    for path in negatives:
        score, negative_score, score_margin = matcher.score_path(path)
        matched = score >= threshold and score_margin >= margin
        rows.append((path.name, score, negative_score, score_margin, matched))

    console.print("\n[bold]Scores[/bold]")
    for name, score, negative_score, score_margin, matched in rows:
        label = "MATCH" if matched else "miss"
        neg_text = f"{negative_score:.3f}" if negative_score is not None else "none"
        console.print(
            f"  {label:5} score={score:.3f} neg={neg_text} "
            f"margin={score_margin:.3f} {name}"
        )

    positive_scores = [row[1] for row in rows[: len(positives)]]
    console.print("\n[bold]Suggested next step[/bold]")
    if not negatives:
        console.print(
            "Record negative samples so Sakura can separate your wake phrase "
            "from room noise and similar wrong phrases:"
        )
        console.print("  sakura wake record-samples --kind negative --count 75")
    else:
        misses = sum(not row[4] for row in rows[: len(positives)])
        false_hits = sum(row[4] for row in rows[len(positives) :])
        console.print(
            f"  positive misses: {misses}/{len(positives)}  "
            f"negative false wakes: {false_hits}/{len(negatives)}"
        )
        console.print(
            "  Tune with SAKURA_WAKE_SAMPLE_THRESHOLD and "
            "SAKURA_WAKE_SAMPLE_MARGIN."
        )
    console.print(
        f"  positive score range: {min(positive_scores):.3f}-"
        f"{max(positive_scores):.3f}"
    )


__all__ = ["wake"]
