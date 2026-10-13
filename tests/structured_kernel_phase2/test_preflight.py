import json
from pathlib import Path
import tarfile

import pytest

from analyse.structured_kernel_phase2.preflight import assess_data_use, validate_resume, make_package, sha, REASON


def test_historical_pool_is_blocked_without_labels():
    result = assess_data_use({'data_use': 'both historical development; B excluded from stage2, not independent confirmation'},
                            {'independent_confirmation': [], 'reason': 'previously used for development', 'use': 'development_only'})
    assert result['state'] == 'BLOCKED'
    assert result['reason_code'] == REASON
    assert result['B_revealed'] is False


def test_missing_metadata_cannot_be_interpreted_as_independence():
    assert assess_data_use({}, {})['reason_code'] == 'DATA_USE_METADATA_MISSING'


def test_changed_declaration_requires_full_scientific_preflight():
    result = assess_data_use({'data_use': 'independent'}, {'independent_confirmation': ['x'], 'reason': 'new evidence', 'use': 'confirmation'})
    assert result['state'] == 'REQUIRES_FULL_PREFLIGHT'


def test_resume_preserves_stop_and_rejects_identity_or_reveal_changes(tmp_path):
    path = tmp_path / 'status.json'
    status = {'state': 'BLOCKED', 'run_identity': 'frozen', 'B_revealed': False}
    path.write_text(json.dumps(status))
    before = path.read_bytes()
    assert validate_resume(tmp_path, 'frozen') == status
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='identity'):
        validate_resume(tmp_path, 'changed')
    status['B_revealed'] = True
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='unrevealed'):
        validate_resume(tmp_path, 'frozen')


def test_evidence_package_has_verifiable_members_and_no_weights(tmp_path):
    (tmp_path / 'status.json').write_text('{"state":"BLOCKED"}\n')
    make_package(tmp_path)
    receipt = json.loads((tmp_path / 'package_result.json').read_text())
    assert receipt['sha256'] == sha(tmp_path / 'evidence_package.tar.gz')
    with tarfile.open(tmp_path / 'evidence_package.tar.gz') as archive:
        assert archive.getnames() == ['status.json', 'package_manifest.json']
    (tmp_path / 'model.pt').write_text('not a model')
    with pytest.raises(ValueError, match='member'):
        make_package(tmp_path)
