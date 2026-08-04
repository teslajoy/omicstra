# the deck

`omicstra-deck.html` - 13 slides, self-contained, no build step. Arrow keys move,
`N` toggles speaker notes, `P` prints.

It lives in `design/` rather than `docs/` deliberately: `docs/` is what GitHub Pages
publishes, and this is not published.

## the two PDFs

Both are regenerated from the HTML - do not edit them.

| file | what it is |
|---|---|
| `omicstra-deck.pdf` | slides only, 13 pages at 1280x720 |
| `omicstra-deck-presenter.pdf` | the same 13 slides with notes printed underneath, on a taller page |

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# slides only
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=6000 \
  --print-to-pdf="$PWD/design/omicstra-deck.pdf" \
  "file://$PWD/design/omicstra-deck.html"

# presenter: notes on, slide height released, taller page
sed -e 's|<body>|<body class="notes-on">|' \
    -e 's|height:720px;background:#fff|min-height:720px;background:#fff|' \
    -e 's|@page{size:13.333in 7.5in;margin:0}|@page{size:13.333in 9.6in;margin:0}|' \
    design/omicstra-deck.html > /tmp/deck-presenter.html
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=6000 \
  --print-to-pdf="$PWD/design/omicstra-deck-presenter.pdf" \
  "file:///tmp/deck-presenter.html"
```

Two things the sed handles, both of which produce a broken PDF if skipped:

- **`notes-on`** - `.notes` is `display:none` by default, so without the class the
  presenter PDF is identical to the slides-only one.
- **`min-height`** - slides are a fixed `720px`. With notes appended the content
  overflows onto extra pages: 16 pages instead of 13. Releasing the height and
  giving `@page` room fixes it.

The screen CSS also needs `transform:none` in the print block, because `fit()` writes
an inline `scale()` for the browser and the PDF would otherwise inherit it and print
every slide shrunk into a corner. That is already in the file.

## speaker notes

`omicstra-deck-notes.md` is the long form - per-slide beats, the numbers to have ready
with the artifact each came from, and the questions to expect. The in-deck `N` notes are
the short form of the same thing.

## provenance

Every number in the deck was checked against `runs/tnbc-92_v3/` on 2026-08-04:

| slide | claim | source |
|---|---|---|
| 6 | AUC and ARI for all five plotted runs | `{run}/metrics_h1_ci.json`, `eval/H2/mc_coherence.parquet` |
| 6 | pathways transferring cross-patient, 0-4 per run | `eval/H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet` |
| 7 | the six biology-over-patient ratios | `eval/H2/metrics_h2_ci.json`, block `h2c` |
| 10 | B4 TLS decode z = 25.5 | the published report, H2 part B.2 |
| 10 | mc01 = CAF / desmoplastic stroma | `scripts/extract_mc14_differential_markers.R` |

**The claim to keep honest:** R4 is the only *contrastive* run carrying all four pathways
cross-patient. B1, B2 and B3 also reach 4 of 4 - R4 wins on magnitude, roughly 1.5x on
every pathway. "The only run that transfers" is false and was corrected.