"""Multi-rater reliability: ICC (two-way ANOVA components) and Krippendorff's alpha (interval).

Rows = responses, columns = raters; values = final grades on the 0-5 scale.
"""
from __future__ import annotations

import numpy as np


def anova_components(values: np.ndarray) -> tuple[float, float, float]:
    n_subjects, n_raters = values.shape
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    grand_mean = values.mean()

    ms_rows = (
        n_raters * np.sum((row_means - grand_mean) ** 2) / (n_subjects - 1)
    )
    ms_cols = (
        n_subjects * np.sum((col_means - grand_mean) ** 2) / (n_raters - 1)
    )
    residual = values - row_means[:, None] - col_means[None, :] + grand_mean
    ms_error = np.sum(residual**2) / ((n_subjects - 1) * (n_raters - 1))
    return ms_rows, ms_cols, ms_error


def icc(values: np.ndarray) -> dict[str, float]:
    n_subjects, n_raters = values.shape
    ms_rows, ms_cols, ms_error = anova_components(values)

    icc_2_1 = (ms_rows - ms_error) / (
        ms_rows
        + (n_raters - 1) * ms_error
        + n_raters * (ms_cols - ms_error) / n_subjects
    )
    icc_2_k = (ms_rows - ms_error) / (
        ms_rows + (ms_cols - ms_error) / n_subjects
    )
    icc_3_1 = (ms_rows - ms_error) / (ms_rows + (n_raters - 1) * ms_error)
    icc_3_k = (ms_rows - ms_error) / ms_rows

    return {
        "ICC(2,1)": icc_2_1,
        "ICC(2,k)": icc_2_k,
        "ICC(3,1)": icc_3_1,
        "ICC(3,k)": icc_3_k,
    }


def krippendorff_alpha_interval(values: np.ndarray) -> float:
    do_num = 0.0
    do_den = 0.0

    for row in values:
        valid = row[~np.isnan(row)]
        n_valid = len(valid)
        if n_valid < 2:
            continue
        row_sum_squares = np.sum((valid - valid.mean()) ** 2)
        do_num += 2.0 * n_valid * row_sum_squares
        do_den += n_valid * (n_valid - 1)

    all_values = values[~np.isnan(values)]
    if do_den == 0 or len(all_values) < 2:
        return np.nan

    observed_disagreement = do_num / do_den
    expected_disagreement = (
        2.0
        * np.sum((all_values - all_values.mean()) ** 2)
        / (len(all_values) - 1)
    )
    return 1.0 - observed_disagreement / expected_disagreement
