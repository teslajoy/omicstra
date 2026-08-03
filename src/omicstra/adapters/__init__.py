"""cohort adapters - the one place that knows a file layout.

each adapter produces AnnData in the shape the EDA steps expect, so no step
ever learns a cohort's storage format.
"""
