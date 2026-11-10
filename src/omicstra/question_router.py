"""TF-IDF cosine router: maps free-text questions to task family IDs.

Router protocol: route(question) -> RouteResult
  task_id: matched family, or None when confidence < threshold (refusal)
  confidence: cosine similarity of top match
  options: all families sorted by score descending

swap rule (pre-registered, 2026-11-10):
  bar: >= 90% top-1 on held-out questions, <= 2 out-of-scope questions
       misrouted above threshold.
  if B misses bar: add sentence-transformers as omicstra[router-embed]
  and plug EmbeddingRouter in behind the same Router protocol.
  default install stays tfidf, no transformers pulled in as core.

record: router=tfidf, version=ROUTER_VERSION, threshold on RunRecord.

requires scikit-learn: install omicstra[measure] or omicstra[router].
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

ROUTER_VERSION = "1"
DEFAULT_THRESHOLD = 0.15


class RouteResult(dict):
    """route(question) return value.

    keys: task_id (str | None), confidence (float),
          options (list[{task_id, score}] sorted desc).
    """


@runtime_checkable
class Router(Protocol):
    """plug-in seam. TFIDFRouter and any future EmbeddingRouter implement this."""

    threshold: float
    version: str

    def route(self, question: str) -> RouteResult: ...
    def to_record(self) -> dict: ...


# vocabulary blocks: description + method terms + example questions.
# these are the ONLY questions visible to the router.
# held-out test questions in tests/fixtures/router_questions.json are separate.
FAMILY_BLOCKS: dict[str, str] = {
    "cross_modal_retrieval": """
        Given an H&E histology image patch or niche retrieve the spatially
        co-registered transcriptomics spot or niche that best matches it.
        Evaluates whether aligned embeddings enable paired cross-modal lookup.
        retrieval Recall@K recall at k MRR mean reciprocal rank nearest neighbor
        search cross-modal matching paired lookup rank-based top-k accuracy
        retrieval rank cosine retrieval paired cross-modal query lookup
        find matching spot correct ST spot H&E tile query embedding
        rank top ranked paired match histology patch gene expression profile
        Can I use a histology patch to find its matching gene expression profile?
        How well does the alignment retrieve the correct ST spot given an H&E tile?
        What is the Recall@1 for cross-modal matching in the aligned space?
        Does the alignment enable paired lookup between modalities?
    """,
    "structural_agreement": """
        Measure whether the geometric structure of the H&E embedding space and
        the ST embedding space agree after alignment. Tests whether topology
        distances and neighborhoods are preserved across modalities.
        CKA centered kernel alignment RSA representational similarity analysis
        Procrustes manifold alignment quality structural correspondence geometry
        pairwise distance correlation topology preservation neighborhood structure
        distance matrix geometric agreement structural similarity two spaces
        pairwise distances consistent arrangement
        Do the two embedding spaces have similar structure after alignment?
        Is the geometry of H&E embeddings preserved in the ST space?
        Does alignment maintain neighborhood relationships between modalities?
        Are the two representation spaces geometrically consistent?
    """,
    "tissue_state_grouping": """
        Cluster or group aligned embeddings to see whether they separate known
        tissue states cell types or megacluster labels. Tests whether the shared
        space encodes biologically meaningful groupings.
        ARI adjusted rand index silhouette NMI normalized mutual information
        k-means clustering tissue state cell type megacluster biological grouping
        unsupervised UMAP cluster group subtype niche subtype immune stromal
        tumor microenvironment recover annotated
        Do aligned niches cluster by tumor microenvironment cell type?
        What is the ARI between k-means clusters and megacluster labels?
        Does the aligned space separate stromal from immune niches?
        Can unsupervised clustering recover biological tissue states?
    """,
    "subject_identity_suppression": """
        Test whether patient or sample identity information is suppressed in the
        aligned embedding space. A good alignment preserves biology while removing
        batch and subject-level confounders.
        LISI local inverse Simpson index batch correction kBET adversarial probe
        patient identity subject suppression confounder batch effect mixing iLISI
        cLISI domain adversarial inter-patient residual patient-level clustering
        patient ID sample identity batch mixing
        Can a classifier predict patient ID from aligned embeddings?
        Does the alignment remove inter-patient batch effects?
        How well are subject-level confounders suppressed in the shared space?
        Is there residual patient clustering in the aligned space?
    """,
    "pathway_transfer": """
        Test whether the aligned shared space encodes Reactome biological pathway
        signals. Uses CCA or Pearson correlation between embedding components and
        pathway activity scores to measure how much pathway biology is captured.
        CCA canonical correlation analysis Pearson r Reactome pathway cosine
        gpath2vec AUCell pathway activity pathway transfer pathway signal
        TGF-beta immune ECM cell cycle apoptosis programmed cell death pathway
        score pathway variance correlation pathway content gene program
        Does the aligned space encode TGF-beta signaling information?
        How correlated are embedding dimensions with Reactome pathway scores?
        Is immune pathway activity recoverable from aligned H&E embeddings?
        What fraction of pathway variance is captured by the alignment?
    """,
    "pathway_discrimination": """
        Test whether the aligned space can discriminate between specific biological
        pathways at fine resolution using the Reactome DAG hierarchy. Extends
        pathway transfer to per-pathway specificity across hundreds of DAG nodes.
        DAG directed acyclic graph Reactome DAG pathway discrimination per-pathway
        AUC ROC pathway specificity pathway DAG nodes fine-grained pathway
        IFN regulation IL signaling BTLA checkpoint pathway hierarchy 395 nodes
        one-sided discriminate distinguish specific pathway AUC threshold
        Can the alignment distinguish IFN regulation from IL signaling pathways?
        What is the AUC for discriminating specific Reactome DAG nodes?
        Does R4 discriminate BTLA checkpoint from ECM collagen biology?
        Can the model tell apart individual pathways in the Reactome hierarchy?
    """,
    "morphology_decode": """
        Test whether morphological features such as nuclei density cell size
        staining intensity and texture can be decoded from ST embeddings via
        linear regression or probe. Measures how much morphological information
        is in the transcriptomic representation.
        linear probe morphology prediction decode R-squared morphological features
        nuclei density cell size staining intensity texture regression
        ST-to-morphology spatial transcriptomics morphology prediction
        decode morphology from expression predict from ST embeddings
        Can nuclei density be predicted from ST spot embeddings?
        How much morphological information is in the transcriptomic space?
        What is the R² for decoding cell size from Novae embeddings?
        Can H&E morphology features be recovered from gene expression?
    """,
}


class TFIDFRouter:
    """TF-IDF cosine similarity router over task family vocabulary blocks.

    fit on FAMILY_BLOCKS at instantiation. route(question) returns the
    best-matching family and confidence score.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self.version = ROUTER_VERSION
        self._family_ids: list[str] = []
        self._vectorizer = None
        self._matrix = None
        self._fit()

    def _fit(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "TFIDFRouter requires scikit-learn: "
                "pip install 'omicstra[measure]'"
            ) from exc

        self._family_ids = list(FAMILY_BLOCKS.keys())
        docs = [FAMILY_BLOCKS[fid] for fid in self._family_ids]
        self._vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(docs)

    def route(self, question: str) -> RouteResult:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(
            zip(self._family_ids, sims.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        top_id, top_score = ranked[0]
        return RouteResult(
            task_id=top_id if top_score >= self.threshold else None,
            confidence=round(top_score, 4),
            options=[{"task_id": fid, "score": round(s, 4)}
                     for fid, s in ranked],
        )

    def to_record(self) -> dict:
        return {
            "router": "tfidf",
            "version": self.version,
            "threshold": self.threshold,
        }