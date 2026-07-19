"""Feature definitions shared by training and inference."""

from __future__ import annotations

import pandas as pd

CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]


import pandas as pd


def add_engagement_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement-related features from raw online shopping session data.

    Features Created
    ----------------
    1. Total_PageViews
       Total number of pages viewed during the session.

    2. Total_Duration
       Total time spent across all page categories.

    3. Avg_Time_Per_Page
       Average time spent per viewed page.

    4. Engagement_Score
       Overall engagement based on both page views and time spent.

    5. Product_Page_Ratio
       Fraction of viewed pages that are product-related.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw dataframe containing the original Online Shoppers dataset.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with the engineered features added.
    """

    result = frame.copy()

    # -------------------------------------------------------
    # Aggregate Features
    # -------------------------------------------------------

    result["Total_PageViews"] = (
        result["Administrative"]
        + result["Informational"]
        + result["ProductRelated"]
    )

    result["Total_Duration"] = (
        result["Administrative_Duration"]
        + result["Informational_Duration"]
        + result["ProductRelated_Duration"]
    )

    # -------------------------------------------------------
    # Derived Features
    # -------------------------------------------------------

    result["Avg_Time_Per_Page"] = (
        result["Total_Duration"]
        / (result["Total_PageViews"] + 1)
    )

    result["Engagement_Score"] = (
        result["Total_PageViews"]
        * result["Total_Duration"]
    )

    result["Product_Page_Ratio"] = (
        result["ProductRelated"]
        / (result["Total_PageViews"] + 1)
    )

    return result


def encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's one-hot encoding and boolean conversion."""
    encoded = pd.get_dummies(
        frame, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=int
    )
    if "Weekend" in encoded:
        encoded["Weekend"] = encoded["Weekend"].astype(int)
    if "Revenue" in encoded:
        encoded["Revenue"] = encoded["Revenue"].astype(int)
    return encoded
