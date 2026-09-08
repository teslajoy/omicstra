# adding a modality

The contract a new modality generates against. Written so it can be followed
without reading the code: everything below is a declaration you make or an
artifact you produce, and the checks that will be run against them are already
in the package.

A modality is **not** a new pipeline. It is an encoder, a unit, an adapter, and
a set of bindings. If adding one requires editing a graph, the design has
failed and that is a bug worth reporting.

---

## 0 · what "a modality" means here

```
a MODALITY is    one assay, one frozen encoder, one unit of observation
it supplies      an adapter, a platform declaration, and role bindings
it does NOT      add a graph node, a gate, or a branch anywhere
```

The existing two are H&E morphology (a vision encoder over image tiles) and
spatial transcriptomics (a graph encoder over expression). A third might be
protein abundance, chromatin, or electron microscopy. The shape is the same.

---

## 1 · the eight roles

Every cohort binds its artifacts to these. They live in
`src/omicstra/configs/data_contract.json` under `roles.definitions`, and the
inventory protocol's `bind` step reports which are satisfied.

| role | required | what it is |
|:--|:--|:--|
| `key` | yes | the identifier joining a measurement to its position and to the image. **May be composite** - per-sample-unique is not cohort-unique |
| `molecular` | yes | the raw integer count matrix. Bound only when integrality is **tested** |
| `platform` | yes | per-sample declaration. Never inferred |
| `spatial` | yes | position per unit, plus the scale that converts it to microns |
| `confounder_axis` | yes | the variable held out when splitting. Declared in `cohort.json` |
| `image` | no | the stained image and its mapping into the key's system |
| `annotation` | no | expert labels per unit, with coverage and granularity |
| `supervision` | no | any signal used to define labels or positive pairs. **Bound separately from `molecular` on purpose** |

**A required role left unbound is a halt, not a caution.** A protocol reading an
unbound role would be inventing data.

---

## 2 · what you must declare

Five things, from `data_contract.json#evidence_contract.required`:

### `platform.json` - per sample, never inferred

```json
{
  "samples":   { "SAMPLE_A": "my_platform" },
  "platforms": {
    "my_platform": {
      "name": "...",
      "spot_pitch_um": 150,
      "spot_diameter_um": 100,
      "resolution_class": "spot | subcellular | single_cell",
      "transcriptome_scope": "whole_transcriptome | targeted_panel",
      "same_section_as_image": true,
      "position_columns": ["pixel_x", "pixel_y"]
    }
  }
}
```

Three traps, each of which has already caught someone:

- **pitch is not diameter.** Pitch is centre-to-centre, diameter is the capture
  footprint. Confusing them mis-scales every micron-denominated figure.
- **platform cannot be sniffed from the object.** A public archive stores
  subcellular imaging data in a spot-shaped schema, so column-sniffing reports
  a cell platform as a spot one, confidently. Declare it.
- **a cohort may be heterogeneous.** Platform is a property of a **sample**, not
  of a cohort. `samples` is a flat map for exactly that reason.

### `cohort.json` - what a person declares about the cohort

```json
{
  "subject_id_column": "patient_id",
  "data_classification": "public | restricted",
  "compute_backend": "mac | slurm | arc",
  "model_backend": "anthropic | bedrock"
}
```

`data_classification` **defaults to restricted**, so forgetting it fails closed.
Public additionally requires provenance - a redistribution claim with no source
is not checkable.

### an adapter

`src/omicstra/adapters/<name>.py`, mapping your files to the `raw_counts`
schema in `data_contract.json#artifacts`. This is the only place that learns a
file format. Every measure downstream receives an AnnData and never knows where
it came from.

### the unit

Its `k` or radius, **and the pitch it is relative to**. `k=6` spans ~0.5 mm at
150 µm pitch and ~0.33 mm at 100 µm; the number alone means nothing.

### `registration_error_um`

Only when `same_section_as_image` is false. It propagates into every downstream
similarity, and evidence collected at error zero does not transfer unwidened.

---

## 3 · the encoder

Frozen. Nothing in this pipeline fine-tunes one, and a fine-tuned encoder is a
different artifact that invalidates every cached embedding below it.

Declare it in `project.json`:

```json
{ "name": "my_encoder", "dim": 512, "role": "st", "raw": false }
```

**Compatibility is measured, never assumed.** If the encoder's training
distribution does not cover your platform, that is an empirical question with an
answer, not a caveat to write down. The compute contract's
`encoder_compatibility` gate runs a subject-grouped linear probe on a sample
before any full extraction, and demotes to a declared fallback if it does not
clear chance.

Declare that fallback. `cohort.json#encoder_fallback` is what turns a failed
probe from a halt into a recorded demotion.

---

## 4 · what you get for free

Nothing below needs writing or wiring:

- **the inventory protocol** - files, platform, shape, coords, join key, counts,
  funnel, conformance bind
- **the EDA protocol** - ten checks, each reporting `pass`, `fail`, `not_run` or
  `not_applicable` with a reason
- **the admissibility gate** - `proceed | caution | stop`, with the caution path
  pausing to ask a person and carrying its full record set into the question
- **the record ledger** - every emission typed and attributed, so
  `Counter(r.actor for r in records)` is the reproducibility claim as a number
- **applicability** - a check that cannot apply to your platform reports
  `not_applicable` **with the reason**, rather than silently passing

---

## 5 · what you must not do

- **do not use the same signal as input and supervision.** That is why
  `supervision` is a separate role. A result whose label was derived from its
  own input is unfalsifiable.
- **do not split below the confounder axis.** Consecutive sections of one block
  are near-duplicates; splitting below subject level leaks.
- **do not trust a filename.** A directory called "raw counts" holding
  normalised floats is a real thing that has happened. Integrality is tested.
- **do not infer platform, classification, or the subject column.** All three
  are declared. A wrong guess in the permissive direction is the failure each
  declaration exists to prevent.

---

## 6 · adding it

```
1  write the adapter                 files -> AnnData conforming to raw_counts
2  declare platform.json             per sample
3  declare cohort.json               subject column, classification, backends
4  add the encoder to project.json   name, dim, role, and its fallback
5  omicstra eda --project-dir <your cohort>
```

Step 5 runs the whole front half. It will tell you which roles are unbound,
which checks do not apply and why, and whether the data is admissible. **If it
asks you to edit a graph, that is a bug.**

What it will *not* do is produce an evidence pack. That comes from evaluating an
alignment grid, and until a cohort has one it is **not routable** - the router
says so rather than inheriting another cohort's winner. See the README on how
the pack is produced.

---

## 7 · a worked example

The seed cohort is the reference implementation. Read in this order:

```
projects/tnbc-92/platform.json      281 samples, one platform, pitch declared
projects/tnbc-92/cohort.json        subject column, classification, provenance
projects/tnbc-92/inventory.json     what the inventory protocol produced
src/omicstra/adapters/wang_st.py    the adapter
```

And for the heterogeneous case - three platforms in one cohort, which is where
per-sample declaration stops being pedantry - the second cohort's `platform.json`
declares 108 spot-based, 8 spot-based at a different pitch, and 9 subcellular
samples in one file.