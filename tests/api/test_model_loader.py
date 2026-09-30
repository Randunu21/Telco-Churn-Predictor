import pytest

from telco_churn.api.model_loader import (
    UNKNOWN,
    ParsedModelUri,
    _run_id_from_runs_uri,
    parse_model_uri,
)


def test_alias_uri():
    assert parse_model_uri("models:/telco-churn@champion") == ParsedModelUri(
        name="telco-churn", alias="champion"
    )


def test_version_uri():
    assert parse_model_uri("models:/telco-churn/3") == ParsedModelUri(
        name="telco-churn", version="3"
    )


@pytest.mark.parametrize(
    "uri", ["models:/my_model-v2@champion", "models:/my_model-v2/12"]
)
def test_names_with_hyphens_and_underscores(uri):
    assert parse_model_uri(uri).name == "my_model-v2"


@pytest.mark.parametrize(
    "uri", ["runs:/abc123/model", "/opt/models/churn", "./model", "file:///opt/model"]
)
def test_non_registry_uris_have_no_registry_info(uri):
    assert parse_model_uri(uri) == ParsedModelUri()


@pytest.mark.parametrize(
    "uri",
    [
        "models:/telco-churn",  # neither alias nor version
        "models:/telco-churn@",  # empty alias
        "models:/@champion",  # empty name
        "models:/telco-churn/Production",  # deprecated stage URI
    ],
)
def test_malformed_registry_uris_raise(uri):
    with pytest.raises(ValueError):
        parse_model_uri(uri)


def test_run_id_extracted_from_runs_uri():
    assert _run_id_from_runs_uri("runs:/abc123/model") == "abc123"


def test_run_id_unknown_for_local_path():
    assert _run_id_from_runs_uri("/opt/models/churn") == UNKNOWN