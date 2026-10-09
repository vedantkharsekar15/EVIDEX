import json
import pytest
from evaluation.run_eval import main
import os
import sys
from unittest.mock import patch

def test_evaluation_data_format():
    # Verify synthetic claims have expected format
    claims_path = "tests/fixtures/eval/synthetic_claims.json"
    assert os.path.exists(claims_path)
    with open(claims_path, "r") as f:
        claims = json.load(f)
    for c in claims:
        assert "claim_id" in c
        assert "text" in c

    labels_path = "tests/fixtures/eval/synthetic_labels.json"
    assert os.path.exists(labels_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)
    for k, v in labels.items():
        assert isinstance(k, str)
        assert isinstance(v, list)

@patch("sys.argv", ["run_eval.py", "--synthetic"])
def test_synthetic_evaluation_runs_without_error(capsys):
    # This acts as an integration test for the metric reporting 
    # and ensures synthetic demo runs correctly.
    try:
        main()
    except Exception as e:
        pytest.fail(f"Synthetic evaluation failed: {e}")
        
    captured = capsys.readouterr()
    assert "Aggregate Test Metrics" in captured.out
    
    # Assert missing labels logic doesn't crash and reports correctly
    assert "eval_claim_3" in captured.out
    assert "WARNING: No relevant labels available" in captured.out
    assert "Skipped 1 claims due to missing labels: ['eval_claim_3']" in captured.out
    
@patch("sys.argv", ["run_eval.py"])
@patch("evaluation.run_eval.load_artifact")
def test_real_evaluation_handles_missing_corpus(mock_load_artifact, capsys):
    mock_load_artifact.return_value = None
    main()
    captured = capsys.readouterr()
    assert "Error: Real corpus records not found" in captured.out

def test_held_out_selection():
    # Test the deterministic split logic in run_eval.py via importing or mocking
    # Since it's inside main(), we can test it by running with synthetic claims
    # but overriding the claims mock.
    pass

