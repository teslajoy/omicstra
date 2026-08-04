# differential markers per mc_megacluster, from Wang's km14 consensus.
#
# usage: Rscript scripts/extract_mc14_differential_markers.R
# output: data/embeddings/biological_signals/mc14_differential_markers.csv
#
# why this exists: mc14_top_features.csv ranks genes by ABSOLUTE centroid
# loading, which returns MT-CO2 / GAPDH / ACTB / RPL* for nearly every class -
# 54 unique genes across all 14 lists. those are the genes that are high
# everywhere, so they identify nothing. the ranking that actually names a class
# is DIFFERENTIAL: a gene's loading in mc X minus its mean loading across the
# other 13. this script is that ranking, committed, because the identities it
# produces are quoted in the report.

km_path  <- "data/inputs/clustering/Kmeans MC/km14.RDS"
out_csv  <- "data/embeddings/biological_signals/mc14_differential_markers.csv"
top_n    <- 15

km <- readRDS(km_path)
stopifnot(inherits(km, "kmeans"))

centers <- km$centers                      # 14 mc x 3888 genes
n_mc <- nrow(centers)
cat(sprintf("km14: %d classes x %d genes, %d prototypes\n",
            n_mc, ncol(centers), length(km$cluster)))

rows <- list()
for (i in seq_len(n_mc)) {
  # differential loading: this class minus the mean of the other 13
  others <- colMeans(centers[-i, , drop = FALSE])
  diff   <- centers[i, ] - others
  ord    <- order(diff, decreasing = TRUE)[seq_len(top_n)]
  rows[[i]] <- data.frame(
    mc            = i,
    rank          = seq_len(top_n),
    gene          = colnames(centers)[ord],
    diff_loading  = unname(diff[ord]),
    abs_loading   = unname(centers[i, ord]),
    n_prototypes  = km$size[i],
    stringsAsFactors = FALSE
  )
  cat(sprintf("mc%02d (n=%3d): %s\n", i, km$size[i],
              paste(colnames(centers)[ord], collapse = ", ")))
}

out <- do.call(rbind, rows)
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("\nwrote %s (%d rows)\n", out_csv, nrow(out)))
