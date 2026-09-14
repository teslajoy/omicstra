"""upload a staged directory to Zenodo as a new, UNPUBLISHED deposition.

the web uploader is why this exists. it is fine for a paper PDF and unreliable
past a few GB - no resume, no checksum, and a browser tab that must stay open -
which is how a 15 GB deposit fails halfway with nothing to show for it. the API
takes files one PUT at a time, so a failure costs one file rather than the run.

it never publishes. publishing on Zenodo is irreversible: the files of a
published record cannot be changed, only superseded by a new version. so this
leaves the deposition in draft and prints the link for a person to review and
publish, which is the same rule the promote step follows for the evidence pack.

    export ZENODO_TOKEN=...            # or put it in .env
    python scripts/zenodo_deposit.py --stage /path/to/staged --metadata meta.json
    python scripts/zenodo_deposit.py --stage ... --deposition 123456   # resume
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import click
import requests

API = "https://zenodo.org/api"


def _token() -> str:
    """env first, then .env. never echoed, never logged, never in a URL."""
    t = os.environ.get("ZENODO_TOKEN", "").strip()
    if not t:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("ZENODO_TOKEN="):
                    t = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not t:
        raise SystemExit(
            "no ZENODO_TOKEN. create one at\n"
            "  https://zenodo.org/account/settings/applications/tokens/new\n"
            "with scopes deposit:write and deposit:actions, then either export it "
            "or add ZENODO_TOKEN=... to .env (which is gitignored).")
    return t


def _sha256(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@click.command()
@click.option("--stage", required=True, type=click.Path(exists=True, path_type=Path),
              help="directory whose files become the deposition.")
@click.option("--metadata", type=click.Path(exists=True, path_type=Path),
              help="json with a top-level 'metadata' object. required for a new deposition.")
@click.option("--deposition", type=int, default=None,
              help="resume into an existing draft instead of creating one.")
@click.option("--dry-run", is_flag=True, help="show what would upload, touch nothing.")
def main(stage: Path, metadata: Path | None, deposition: int | None, dry_run: bool) -> None:
    """stage a directory into a Zenodo draft. does not publish."""
    files = sorted(p for p in stage.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    click.echo(f"{len(files)} file(s), {total / 1e9:.2f} GB")
    for p in files:
        click.echo(f"  {p.name:52} {p.stat().st_size / 1e6:9.1f} MB")

    if dry_run:
        click.echo("\ndry run - nothing sent.")
        return

    tok = _token()
    s = requests.Session()
    # the token travels in a header, never a query string: URLs land in logs,
    # proxies and shell history, and a leaked deposit token can rewrite a record.
    s.headers.update({"Authorization": f"Bearer {tok}"})

    if deposition:
        r = s.get(f"{API}/deposit/depositions/{deposition}", timeout=60)
        r.raise_for_status()
        dep = r.json()
        click.echo(f"\nresuming draft {deposition}")
    else:
        if not metadata:
            raise SystemExit("--metadata is required to create a new deposition")
        meta = json.loads(metadata.read_text())
        r = s.post(f"{API}/deposit/depositions", json=meta, timeout=60)
        if not r.ok:
            raise SystemExit(f"create failed: {r.status_code} {r.text[:500]}")
        dep = r.json()
        click.echo(f"\ncreated draft {dep['id']}")

    bucket = dep["links"]["bucket"]
    already = {f["filename"]: f.get("checksum", "").removeprefix("md5:")
               for f in dep.get("files", [])}

    for p in files:
        if p.name in already:
            click.echo(f"  {p.name}: already on the draft, skipped")
            continue
        size = p.stat().st_size
        click.echo(f"  {p.name}: uploading {size / 1e6:.1f} MB ...", nl=False)
        with p.open("rb") as fh:
            # streamed, so a 13 GB file never sits in memory
            r = s.put(f"{bucket}/{p.name}", data=fh, timeout=None)
        if not r.ok:
            click.echo(" FAILED")
            raise SystemExit(f"{p.name}: {r.status_code} {r.text[:400]}\n"
                             f"re-run with --deposition {dep['id']} to resume; "
                             "files already uploaded are skipped.")
        click.echo(" ok")

    click.echo(f"\ndraft ready, NOT published:\n  https://zenodo.org/uploads/{dep['id']}")
    click.echo("\nreview it, then publish from that page. publishing is irreversible - "
               "a published record's files cannot be changed, only superseded.")


if __name__ == "__main__":
    sys.exit(main())
