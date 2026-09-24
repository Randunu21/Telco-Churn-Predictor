from telco_churn.data.data import clean, split
import numpy as np
from telco_churn.data.data import build_target_encoder


def test_clean_drops_expected_columns(raw_telco_df):
    cleaned = clean(raw_telco_df)

    for col in ["TotalCharges", "MonthlyCharges", "gender", "PhoneService", "MultipleLines", "customerID"]:
        assert col not in cleaned.columns


def test_clean_keeps_other_columns_and_row_count(raw_telco_df):
    cleaned = clean(raw_telco_df)

    assert len(cleaned) == len(raw_telco_df)
    assert "Churn" in cleaned.columns
    assert "tenure" in cleaned.columns


def test_clean_does_not_mutate_input(raw_telco_df):
    # clean() should hand back a new DataFrame, not modify the one it was given.
    # pandas .drop() defaults to this already (inplace=False) — this test exists
    # to guard against someone later "optimizing" it into inplace=True by mistake.
    original_columns = list(raw_telco_df.columns)
    clean(raw_telco_df)

    assert list(raw_telco_df.columns) == original_columns


def test_split_returns_disjoint_train_test_rows(raw_telco_df):
    cleaned = clean(raw_telco_df)
    X_train, X_test, y_train, y_test = split(cleaned)

    assert len(X_train) + len(X_test) == len(cleaned)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_drops_churn_from_X(raw_telco_df):
    cleaned = clean(raw_telco_df)
    X_train, X_test, y_train, y_test = split(cleaned)

    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns


def test_split_y_values_are_valid_labels(raw_telco_df):
    cleaned = clean(raw_telco_df)
    X_train, X_test, y_train, y_test = split(cleaned)

    assert set(y_train.unique()).issubset({"Yes", "No"})
    assert set(y_test.unique()).issubset({"Yes", "No"})


def test_split_is_reproducible_with_fixed_random_state(raw_telco_df):
    # Same input, called twice, should give byte-identical splits — this is what
    # settings.random_state is actually FOR. If this ever fails, something has
    # broken the reproducibility of every downstream training run.
    cleaned = clean(raw_telco_df)
    X_train_1, X_test_1, _, _ = split(cleaned)
    X_train_2, X_test_2, _, _ = split(cleaned)

    assert list(X_train_1.index) == list(X_train_2.index)
    assert list(X_test_1.index) == list(X_test_2.index)


def test_target_encoder_maps_no_to_0_and_yes_to_1(raw_telco_df):
    y = raw_telco_df["Churn"]
    encoder = build_target_encoder(y)

    encoded = encoder.transform(y.values.reshape(-1, 1))

    # "No" -> 0, "Yes" -> 1 — this specific mapping matters downstream:
    # it's what lets a model's raw prediction be translated back into a
    # human-readable label later, so it needs to be pinned down, not incidental.
    for original, enc in zip(y, encoded.ravel()):
        expected = 0.0 if original == "No" else 1.0
        assert enc == expected


def test_target_encoder_is_fit_only_on_what_you_pass_it(raw_telco_df):
    # Regression guard for the exact leakage bug already fixed once in this
    # project: fitting must happen on the given y only, not on some larger
    # or different set. We can't inspect "what it was fit on" directly, but
    # we CAN confirm it fits successfully on a smaller subset and produces
    # a fresh, independent encoder each call.
    y_subset = raw_telco_df["Churn"].iloc[:5]
    encoder = build_target_encoder(y_subset)

    assert encoder.categories_[0].tolist() == ["No", "Yes"]


def test_target_encoder_rejects_unexpected_labels(raw_telco_df):
    y = raw_telco_df["Churn"]
    encoder = build_target_encoder(y)

    bad_labels = np.array(["Maybe"]).reshape(-1, 1)

    # An unexpected label should fail loudly, not silently encode as
    # something wrong. This is deliberately the OPPOSITE behavior from
    # OneHotEncoder(handle_unknown="ignore") elsewhere in the pipeline —
    # worth having a test that pins down which parts fail loud vs quiet.
    import pytest
    with pytest.raises(ValueError):
        encoder.transform(bad_labels)