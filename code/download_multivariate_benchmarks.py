"""
download_multivariate_benchmarks_v4.py
======================================

Download mainstream multivariate forecasting benchmarks, including the exact
six benchmark families used in the ICLR 2023 Crossformer paper.

Default set:
    ETTh1, ETTh2, ETTm1, ETTm2,
    weather (21-dim Autoformer/TSLib version),
    ILI,
    WTH (12-dim Informer/Crossformer version),
    ECL (321-dim Crossformer version),
    Traffic (862-dim)

Important:
- Crossformer used ETTh1, ETTm1, WTH, ECL, ILI, Traffic.
- Crossformer's WTH is the *12-dimensional Informer WTH*, NOT the later
  21-dimensional Autoformer/Time-Series-Library weather.csv.
- Exchange Rate is intentionally omitted.

Examples
--------
Download the full default set:
    python download_multivariate_benchmarks_v4.py

Download exactly the Crossformer paper benchmarks:
    python download_multivariate_benchmarks_v4.py --preset crossformer

Download a compact set without ECL/Traffic:
    python download_multivariate_benchmarks_v4.py --preset compact

Download selected datasets:
    python download_multivariate_benchmarks_v4.py --datasets ETTh1 WTH ILI

Choose another output directory:
    python download_multivariate_benchmarks_v4.py --output-dir data/benchmarks

Notes
-----
- Uses only Python standard library.
- Tries multiple public mirrors when available.
- Existing files are kept unless --force is supplied.
- Performs basic CSV header/shape validation.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Source:
    url: str
    zip_member: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    sources: tuple[Source, ...]
    expected_rows: Optional[int]
    expected_features: Optional[int]
    description: str
    crossformer: bool = False


REGISTRY: dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec(
        name="ETTh1",
        filename="ETTh1.csv",
        sources=(
            Source(
                "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
                "ETT-small/ETTh1.csv"
            ),
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/ETT-small/ETTh1.csv?download=true"
            ),
        ),
        expected_rows=17420,
        expected_features=7,
        description="ETT hourly subset 1; 7 channels.",
        crossformer=True,
    ),
    "ETTh2": DatasetSpec(
        name="ETTh2",
        filename="ETTh2.csv",
        sources=(
            Source(
                "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
                "ETT-small/ETTh2.csv"
            ),
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/ETT-small/ETTh2.csv?download=true"
            ),
        ),
        expected_rows=17420,
        expected_features=7,
        description="ETT hourly subset 2; 7 channels.",
    ),
    "ETTm1": DatasetSpec(
        name="ETTm1",
        filename="ETTm1.csv",
        sources=(
            Source(
                "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
                "ETT-small/ETTm1.csv"
            ),
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/ETT-small/ETTm1.csv?download=true"
            ),
        ),
        expected_rows=69680,
        expected_features=7,
        description="ETT 15-minute subset 1; 7 channels.",
        crossformer=True,
    ),
    "ETTm2": DatasetSpec(
        name="ETTm2",
        filename="ETTm2.csv",
        sources=(
            Source(
                "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
                "ETT-small/ETTm2.csv"
            ),
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/ETT-small/ETTm2.csv?download=true"
            ),
        ),
        expected_rows=69680,
        expected_features=7,
        description="ETT 15-minute subset 2; 7 channels.",
    ),
    "weather": DatasetSpec(
        name="weather",
        filename="weather.csv",
        sources=(
            Source(
                "https://raw.githubusercontent.com/wayne155/"
                "multivariate_timeseries_datasets/main/weather.zip",
                zip_member="weather/weather.csv",
            ),
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/weather/weather.csv?download=true"
            ),
        ),
        expected_rows=52696,
        expected_features=21,
        description=(
            "21-dimensional Autoformer/TSLib Weather benchmark. "
            "This is NOT Crossformer's 12-dimensional WTH."
        ),
    ),
    "WTH": DatasetSpec(
        name="WTH",
        filename="WTH.csv",
        sources=(
            # Original Informer source used by Crossformer.
            Source(
                "https://drive.google.com/uc?export=download&id="
                "1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG"
            ),
        ),
        expected_rows=None,
        expected_features=12,
        description=(
            "12-dimensional Informer weather dataset used by Crossformer."
        ),
        crossformer=True,
    ),
    "ILI": DatasetSpec(
        name="ILI",
        filename="national_illness.csv",
        sources=(
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/illness/national_illness.csv?download=true"
            ),
            Source(
                "https://raw.githubusercontent.com/scalation/data/master/"
                "Influenza/national_illness.csv"
            ),
        ),
        expected_rows=966,
        expected_features=7,
        description="Influenza-like illness (ILI); weekly data, 7 channels.",
        crossformer=True,
    ),
    "ECL": DatasetSpec(
        name="ECL",
        filename="ECL.csv",
        sources=(
            # Original Informer source used by Crossformer.
            Source(
                "https://drive.google.com/uc?export=download&id="
                "1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP"
            ),
            # Same 321-channel electricity family in TSLib format; useful fallback.
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/electricity/electricity.csv?download=true"
            ),
        ),
        expected_rows=None,
        expected_features=321,
        description="Electricity Consuming Load; 321 channels; used by Crossformer.",
        crossformer=True,
    ),
    "Traffic": DatasetSpec(
        name="Traffic",
        filename="traffic.csv",
        sources=(
            Source(
                "https://huggingface.co/datasets/thuml/Time-Series-Library/"
                "resolve/main/traffic/traffic.csv?download=true"
            ),
            Source(
                "https://raw.githubusercontent.com/unit8co/darts/master/"
                "datasets/traffic.csv"
            ),
        ),
        expected_rows=17544,
        expected_features=862,
        description="Traffic occupancy benchmark; 862 sensors; used by Crossformer.",
        crossformer=True,
    ),
}

PRESETS = {
    "compact": ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "ILI"),
    "crossformer": ("ETTh1", "ETTm1", "WTH", "ECL", "ILI", "Traffic"),
    "all": tuple(REGISTRY.keys()),
}

# The user asked to include Crossformer benchmarks in the normal download set.
DEFAULT_DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "weather",
    "ILI",
    "WTH",
    "ECL",
    "Traffic",
)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Rel-ESE benchmark downloader/4.0",
            "Accept": "*/*",
        },
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        downloaded = 0
        with destination.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    print(
                        f"\r      {downloaded / 1024**2:8.1f} MB / "
                        f"{total_bytes / 1024**2:8.1f} MB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r      downloaded {downloaded / 1024**2:8.1f} MB",
                        end="",
                        flush=True,
                    )
    print()


def _materialize_source(source: Source, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ts_benchmark_") as tmp:
        tmpdir = Path(tmp)
        suffix = ".zip" if source.zip_member else ".download"
        downloaded = tmpdir / ("payload" + suffix)

        _download(source.url, downloaded)

        if source.zip_member:
            with zipfile.ZipFile(downloaded, "r") as zf:
                names = zf.namelist()
                member = source.zip_member

                if member not in names:
                    basename = Path(member).name
                    matches = [x for x in names if Path(x).name == basename]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"Could not find {member!r} in ZIP. "
                            f"Matching basenames: {matches[:10]}"
                        )
                    member = matches[0]

                with zf.open(member, "r") as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            shutil.move(str(downloaded), str(destination))


def _looks_like_html(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:512].lower()
    except Exception:
        return False
    return b"<html" in prefix or b"<!doctype html" in prefix


def _validate_csv(path: Path, spec: DatasetSpec) -> tuple[int, int, str]:
    if _looks_like_html(path):
        raise RuntimeError(
            "Downloaded content looks like HTML rather than CSV. "
            "The source may require an interactive confirmation page."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError("Downloaded file is empty.") from exc

        rows = sum(1 for _ in reader)

    if len(header) < 2:
        raise RuntimeError(f"CSV has too few columns: {header}")

    first_col = header[0].strip()
    n_features = len(header) - 1
    warnings = []

    if spec.expected_rows is not None and rows != spec.expected_rows:
        warnings.append(f"rows expected≈{spec.expected_rows}, got={rows}")

    if spec.expected_features is not None and n_features != spec.expected_features:
        warnings.append(
            f"features expected≈{spec.expected_features}, got={n_features}"
        )

    if warnings:
        first_col += " | WARNING: " + "; ".join(warnings)

    return rows, n_features, first_col


def download_dataset(spec: DatasetSpec, output_dir: Path, force: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / spec.filename

    tag = " [Crossformer]" if spec.crossformer else ""
    print("=" * 78)
    print(f"{spec.name}{tag}: {spec.description}")
    print(f"output : {destination}")

    if destination.exists() and not force:
        print("status : already exists (use --force to re-download)")
        rows, n_features, info = _validate_csv(destination, spec)
        print(f"verify : rows={rows}, features={n_features}, first_col={info}")
        return destination

    errors: list[str] = []

    for index, source in enumerate(spec.sources, start=1):
        print(f"source {index}/{len(spec.sources)}: {source.url}")

        try:
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.unlink(missing_ok=True)
            _materialize_source(source, partial)

            # Validate before replacing an existing good file.
            rows, n_features, info = _validate_csv(partial, spec)
            partial.replace(destination)

            print(f"verify : rows={rows}, features={n_features}, first_col={info}")
            print("status : OK")
            return destination

        except Exception as exc:
            errors.append(f"{source.url}: {exc}")
            print(f"failed : {exc}")
            destination.with_suffix(destination.suffix + ".part").unlink(
                missing_ok=True
            )

    raise RuntimeError(
        f"All mirrors failed for {spec.name}:\n  - " + "\n  - ".join(errors)
    )


def _resolve_requested(args: argparse.Namespace) -> list[str]:
    if args.preset:
        return list(PRESETS[args.preset])

    if not args.datasets:
        return list(DEFAULT_DATASETS)

    aliases = {name.lower(): name for name in REGISTRY}
    aliases.update({
        "illness": "ILI",
        "national_illness": "ILI",
        "ecl": "ECL",
        "electricity": "ECL",
        "traffic": "Traffic",
        "wth": "WTH",
        "weather12": "WTH",
        "weather21": "weather",
    })

    requested = []
    for item in args.datasets:
        if item.lower() == "all":
            return list(PRESETS["all"])

        key = item.lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown dataset {item!r}. Available: {', '.join(REGISTRY)}"
            )

        canonical = aliases[key]
        if canonical not in requested:
            requested.append(canonical)

    return requested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download mainstream multivariate forecasting benchmarks, "
            "including the Crossformer ICLR 2023 benchmark suite."
        )
    )

    p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Explicit datasets to download. Use 'all' for every registered dataset.",
    )
    p.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default=None,
        help=(
            "Preset dataset group: compact, crossformer, or all. "
            "If supplied, --datasets is ignored."
        ),
    )
    p.add_argument("--output-dir", default="data")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available datasets:")
        for name, spec in REGISTRY.items():
            mark = "Crossformer" if spec.crossformer else ""
            print(
                f"  {name:<10} -> {spec.filename:<22} "
                f"features={spec.expected_features!s:<4} "
                f"{mark:<11} {spec.description}"
            )
        print()
        print("Presets:")
        for name, values in PRESETS.items():
            print(f"  {name:<12}: {', '.join(values)}")
        return

    try:
        requested = _resolve_requested(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2)

    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Download directory: {output_dir}")
    print(f"Datasets: {', '.join(requested)}")
    print()

    successes = []
    failures = []

    for name in requested:
        try:
            path = download_dataset(REGISTRY[name], output_dir, args.force)
            successes.append(path)
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"ERROR: {name}: {exc}")

    print()
    print("=" * 78)
    print(f"Downloaded/verified: {len(successes)}/{len(requested)}")

    for path in successes:
        print(f"  OK   {path}")

    for name, error in failures:
        print(f"  FAIL {name}: {error}")

    print()
    print("Crossformer paper benchmark set:")
    print("  ETTh1, ETTm1, WTH(12-dim), ECL, ILI, Traffic")

    if "Traffic" in requested:
        print()
        print(
            "NOTE: Traffic has 862 channels => 371,091 unordered pairs. "
            "The current all-pairs Rel-ESE statistical pipeline needs sparse "
            "edge selection before a full-scale run is practical."
        )

    if "ECL" in requested:
        print()
        print(
            "NOTE: ECL has 321 channels => 51,360 unordered pairs. "
            "For the current all-pairs Rel-ESE implementation, use a subset "
            "or sparse graph for preliminary experiments."
        )

    if "WTH" in requested and "weather" in requested:
        print()
        print(
            "NOTE: WTH.csv and weather.csv are intentionally both retained: "
            "Crossformer used the 12-dimensional Informer WTH, while the "
            "modern Autoformer/TSLib weather.csv has 21 dimensions."
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()