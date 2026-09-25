import pandas as pd
import pandera.pandas as pa
import pytest

from telco_churn.data.schema import validate_raw


def test_valid_raw_df_passes(raw_telco_df):
    # Should not raise. If it does, the fixture drifted from the schema,
    # or the schema is stricter than it should be — either way, worth knowing.
    validate_raw(raw_telco_df)


def test_invalid_category_is_rejected(raw_telco_df):
    bad = raw_telco_df.copy()
    bad.loc[bad.index[0], "InternetService"] = "Satellite"

    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(bad)


def test_negative_tenure_is_rejected(raw_telco_df):
    bad = raw_telco_df.copy()
    bad.loc[bad.index[0], "tenure"] = -5

    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(bad)


def test_multiple_failures_are_all_reported_together(raw_telco_df):
    # This is the whole point of lazy=True — confirm it actually collects
    # every failure in one pass rather than stopping at the first.
    bad = raw_telco_df.copy()
    bad.loc[bad.index[0], "InternetService"] = "Satellite"
    bad.loc[bad.index[0], "tenure"] = -5

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_raw(bad)

    failed_columns = set(exc_info.value.failure_cases["column"])
    assert "InternetService" in failed_columns
    assert "tenure" in failed_columns


def test_extra_unexpected_column_does_not_fail(raw_telco_df):
    # strict=False is deliberate — upstream schema drift (a new column
    # appearing) shouldn't break the pipeline the way a WRONG value should.
    extra = raw_telco_df.copy()
    extra["SomeNewColumnFromUpstream"] = "whatever"

    validate_raw(extra)  # should not raise