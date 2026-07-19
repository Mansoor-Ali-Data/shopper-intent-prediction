import pandas as pd

from shopper_intent.features import add_engagement_features, encode_features


def test_engagement_features_match_notebook_formula():
    source = pd.DataFrame({
        "Administrative": [1], "Informational": [2], "ProductRelated": [3],
        "Administrative_Duration": [10.0], "Informational_Duration": [20.0],
        "ProductRelated_Duration": [30.0],
    })

    result = add_engagement_features(source)

    assert result.loc[0, "Total_PageViews"] == 6
    assert result.loc[0, "Total_Duration"] == 60.0
    assert result.loc[0, "Avg_Time_Per_Page"] == 60.0 / 7
    assert "Total_PageViews" not in source


def test_encoding_converts_booleans_and_creates_dummy_columns():
    source = pd.DataFrame({
        "Month": ["Feb"], "VisitorType": ["Returning_Visitor"],
        "OperatingSystems": [1], "Browser": [2], "Region": [1],
        "TrafficType": [3], "Weekend": [True], "Revenue": [False],
    })

    result = encode_features(source)

    assert result.loc[0, "Weekend"] == 1
    assert result.loc[0, "Revenue"] == 0
    assert "Month_Feb" in result.columns
    assert "VisitorType_Returning_Visitor" in result.columns
