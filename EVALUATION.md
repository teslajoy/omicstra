# EVALUATION.md

what each test proves, and the condition under which it proves it.

a catalogue, not a protocol. every row is stated for any cohort and any pair of
modalities. the test and its condition transfer unchanged; a threshold transfers
or does not depending on WHICH KIND it is, which is the distinction section 8
makes and which this document is wrong without.

the form is always the same:

```
TEST            what is computed
PROVES          the claim it licenses
ONLY IF         the condition. fail it and the number still exists,
                but it licenses a different, smaller claim
INSTEAD READS   what the number means when the condition fails
```

that last line is the point of the document. a test whose condition fails rarely
measures nothing - it measures something else, and reporting it as the intended
claim is the failure this catalogue exists to prevent.

---

## 1 · identifiability — do two representations describe the same thing

### paired retrieval

```
TEST            discrimination (AUC, recall@k, MRR) between matched pairs and
                mismatched pairs drawn from the same units
PROVES          the two representations of one unit are mutually identifiable
ONLY IF         pairs are scored across HELD-OUT subjects, and the paired unit
                is the same physical unit in both modalities
INSTEAD READS   within-subject: how distinctive subjects are
                serial sections: how similar adjacent tissue is
```

### geometric agreement

```
TEST            representational similarity between the two spaces (CKA, or a
                before/after alignment gap)
PROVES          the two modalities agree in GEOMETRY, not merely that one
                retrieves the other
ONLY IF         computed on held-out subjects and reported BESIDE retrieval
INSTEAD READS   alone, nothing decisive: a space can retrieve well while
                agreeing poorly, and the reverse
```

---

## 2 · structure — does a space recover biological organisation

**this is the trickiest family, because three different things produce the same
number.** a clustering score can rise from biology, from the label being a proxy
for the subject, or from the geometry of the space itself. the three tests below
separate them and must be read together.

### label granularity, before anything else

```
TEST            normalised mutual information between the candidate label and
                the subject identifier, on the same units the score will use
PROVES          the label is not a restatement of provenance
ONLY IF         run BEFORE any clustering metric is computed or reported
INSTEAD READS   above threshold, every clustering score against that label is a
                subject-classification score wearing a biological name. the
                label stays diagnostic; it cannot carry a biological claim alone
```

### clustering agreement

```
TEST            agreement between unsupervised clusters and a biological label
                (ARI, or an equivalent chance-corrected index)
PROVES          the space organises units by that biology
ONLY IF         the granularity test above passes, AND the compared spaces are
                matched on effective rank
INSTEAD READS   unmatched rank: the higher-dimensional space wins on geometry
                alone. the metric rewards spread, not meaning
```

### decodability, as the companion

```
TEST            a linear probe for the same label, on the same split
PROVES          the information is PRESENT in the space
ONLY IF         the split is the same one the clustering used
INSTEAD READS   a label can be linearly decodable and not clustered. probe high
                with clustering low means the information is there and the
                geometry does not group it - which is a finding, not a failure
```

> **read these three together or not at all.** a clustering number alone cannot
> distinguish "no biology", "biology present but not clustered", and "the label
> was provenance".

---

## 3 · confounding — is it biology, or is it where the sample came from

### the paired z-ratio

```
TEST            on ONE space and ONE test: the permutation z for a biological
                label, divided by the permutation z for the subject identifier
PROVES          the space represents biology more strongly than provenance
ONLY IF         the null permutes the biological label AT SUBJECT LEVEL, both z
                come from the same space and the same test, and the ratio is
                compared against the RAW un-aligned modality's own ratio
INSTEAD READS   permuting within subject tests a different hypothesis and
                inflates the biological z
```

### the anti-helpful finding

```
TEST            the same ratio, against the raw un-aligned floor
PROVES          when a method falls BELOW the floor: it is not the weakest
                option, it is anti-helpful - it amplifies provenance by
                construction
ONLY IF         the floor is the raw form of the SAME modality on the SAME units
INSTEAD READS   a method below the floor that is still ranked among the others
                invites a reader to choose it
```

this is the test most worth running first on any new cohort. it is structural in
any design where several regions come from one subject, which is most spatial
designs: a method can score well by learning the block rather than the biology,
and only a confound diagnostic separates the two.

---

## 4 · generalisation — does it hold on subjects never seen

### held-out subject evaluation

```
TEST            every reported score computed on subjects excluded from training
PROVES          the result is not a restatement of subject identity
ONLY IF         the split is identical across every method compared
INSTEAD READS   a split that varies by method makes the comparison meaningless
                before the metric is chosen
```

### program-level transfer

```
TEST            association between a per-program score and the space, on
                held-out subjects, against a permutation null, corrected across
                the whole family of programs tested
PROVES          a program-level signal carries to subjects never trained on
ONLY IF         the view tested does not itself encode the program being tested
INSTEAD READS   a circular view: the signal is the input, and the test is a
                self-correlation
```

**circularity is derived, never trusted.** it is a property of what a method
received as input, so it is computed from the declared inputs - not read from a
flag in an artifact, which goes stale the moment an arm is added.

---

## 5 · attribution — detecting a signal and naming its source are different claims

### the specificity matrix

```
TEST            the full matrix of scores between every program and every
                program axis, not only the diagonal
PROVES          the method ATTRIBUTES a signal to the right source
ONLY IF         the off-diagonal is reported with the diagonal
INSTEAD READS   high off-diagonal: the method detects that something is there
                and cannot say what. a diagonal reported alone looks identical
                to specificity and is not
```

### resolution

```
TEST            the same specificity test at a finer decomposition of the
                program hierarchy
PROVES          an apparent lack of specificity is real rather than an artifact
                of aggregating programs too coarsely
ONLY IF         the finer decomposition is pre-declared, not chosen after seeing
                the coarse result
INSTEAD READS   aliasing at coarse resolution can be a property of the
                aggregation, not of the method
```

---

## 6 · floors and nulls — what the number must beat

```
TEST            an untrained or randomly-initialised control arm, scored
                identically
PROVES          training created the signal
ONLY IF         the control differs from the trained arm ONLY in training
INSTEAD READS   a signal the untrained projection preserves was already in the
                encoder, not created by alignment
```

```
TEST            the STRONGER single modality, scored on the same units
PROVES          fusion adds something neither modality had alone
ONLY IF         the comparison is against the stronger of the two
INSTEAD READS   against the weaker modality the gain is overstated, sometimes by
                several fold
```

```
TEST            a permutation null for the statistic, reported as z or p
PROVES          the magnitude is not what the null already produces
ONLY IF         the verdict uses the z or the p
INSTEAD READS   a raw magnitude against an eyeballed bar has a null it computed
                and ignored. the permutation count then changes the runtime and
                nothing else
```

```
TEST            spatial autocorrelation against a permutation null
PROVES          a measurement has spatial structure at all - which decides
                whether a spatial model is warranted over a non-spatial one
ONLY IF         the criterion uses the null, not a magnitude threshold
INSTEAD READS   a magnitude bar is a cohort-calibrated choice and does not
                transfer to a new cohort
```

### a rank-based score has a denominator, and filtering changes it

```
TEST            a rank-based enrichment score (AUC over the ranking of all
                measured features in a unit) against a declared gene set
PROVES          relative pathway activity, comparably ACROSS units
ONLY IF         every unit is ranked over the SAME feature universe
INSTEAD READS   restrict the input per unit - to its variable features, its
                detected features, its top-N - and each unit is scored against
                a different denominator. the scores stop being comparable
                between units, which is the one thing they were for
```

the tempting version of this error is filtering to "the interesting genes"
first. it feels like noise reduction and it silently makes every unit's score
incomparable to every other unit's, because each unit chose its own filter.

a selection-based test is the opposite and wants the filter: a hypergeometric
or Fisher enrichment asks whether a CHOSEN set is over-represented against a
stated background, so the selection is the question rather than a corruption of
it. the two cannot share a preprocessing path, and a pipeline that gives them
one has broken the rank-based half.

---

## 7 · coverage — is the comparison complete

```
TEST            which arms of the declared grid each rollup actually scored
PROVES          the comparison covers what it claims to compare
ONLY IF         the grid is declared in advance
INSTEAD READS   a missing arm is a gap in a comparison, not an absence of
                signal - and the arm most often missing is the control
```


---

## 8 · thresholds — three kinds, and only one of them is free

a threshold is not one thing, and treating it as one is how a criterion with no
external basis ends up travelling to a cohort it was never derived on.

### a statistical standard

```
EXAMPLES        a significance level; a multiple-testing correction applied
                across a declared family; a confidence-interval coverage; a
                |z| cut implied by one of those
TRANSFERS       YES. these are conventions about error rates, not about tissue,
                and they are the field's, not a cohort's
STILL DECLARE   which level, which correction, and over WHICH FAMILY. the family
                is the part that silently changes between cohorts: the same
                correction over a different number of tests is a different test
```

### a cut derived from this cohort's own null

```
EXAMPLES        a permutation null; a bootstrap interval; an empirical p; a
                floor read from the raw un-aligned modality on these units
TRANSFERS       the METHOD transfers, the NUMBER does not. a new cohort
                re-derives it from its own data, which is what makes it
                principled rather than inherited
STILL DECLARE   the null's construction - what is permuted, at what level, how
                many times - because that, not the resulting number, is the
                thing a reader has to be able to reproduce
```

### a bar somebody chose by looking

```
EXAMPLES        a magnitude cut on a raw statistic; a count of markers that must
                clear it; any "> x" whose x has no citation and no null behind it
TRANSFERS       NO. it encodes the cohort it was chosen on, and it will be
                roughly right on similar tissue and silently wrong on different
                tissue
STILL DECLARE   that it IS one. mark it calibrated-here, and have the gate
                ESCALATE on a new cohort rather than reuse it. a calibrated bar
                that is honest about being calibrated is usable; one that is not
                is the most portable kind of error in this field
```

**the test to apply to your own criterion:** can you name the external basis, or
the null it came from? if neither, it is the third kind, whatever it looks like.
a magnitude bar in the position where the literature uses a corrected p is the
common case, and the tell is that the null was computed and then discarded.

---

## the rule that binds them

**admissibility is task-scoped.** a method contraindicated for one question may
be the right answer to another, and a contraindication is never a verdict on a
method. so a result is reported as an entry in a question x method table - a value,
whether the method is admissible for THAT question, and why. ("cell" for such an
entry is the usual word and the wrong one in this field; nothing here is a
biological cell.) a value with no admissibility beside it is an invitation to
compare things that are not comparable.

**a threshold transfers according to its kind - see section 8.** a new cohort
inherits every form above; which numbers come with it depends on whether each
threshold is a standard, a null-derived cut, or a bar somebody eyeballed.

**a guard that did not run is not a guard that passed.** a hypothesis whose
required guards all lacked an input reports no failure, and that reads as a clean
result for a number nothing checked. it is reported `unguarded`, which is a
different statement from `supported`.
