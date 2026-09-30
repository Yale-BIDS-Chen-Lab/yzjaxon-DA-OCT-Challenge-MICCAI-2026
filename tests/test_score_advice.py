"""Scoring reports reproduction differences without treating them as success."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import repro_score as S


def test_measured_score_with_different_environment_and_weights_is_not_a_pass(tmp_path, monkeypatch):
    a, b = tmp_path / 'a.pt', tmp_path / 'b.pt'
    a.write_bytes(b'a')
    b.write_bytes(b'b')
    expected = {
        'synthetic_reference': {'labelled_images': 1, 'environment': {'gpu': 'reference'},
                               'final_score': 0.3, 'macula_score': 0.6, 'widefield_score': 0.0},
        'published_weights': {'model.pt': {'sha256': 'different'},
                              'model_b.pt': {'sha256': 'different'}}}
    monkeypatch.setattr(S, '_expected', lambda: expected)
    monkeypatch.setattr(S, '_environment', lambda: {'gpu': 'custom'})
    for name in ('scoring.py', 'metrics.py'):
        (tmp_path / name).write_text('# local\n')
    expected['needs'] = {'kit': {'files': {
        'starting_kit/app_scoring/program/' + name: {'sha256': 'reference'}
        for name in ('scoring.py', 'metrics.py')}}}
    monkeypatch.setattr(S, '_official_scorer', lambda *args: tmp_path / 'scoring.py')
    monkeypatch.setattr(S.paths, 'RUNS_DIR', tmp_path / 'runs')
    def layout(root, source, count):
        images, masks = root / 'images', root / 'masks'
        images.mkdir()
        masks.mkdir()
        (root / 'input/res').mkdir(parents=True)
        (masks / 'one.png').write_bytes(b'mask')
        return images, masks, [('one.png', 'one.png')]
    monkeypatch.setattr(S, '_layout', layout)
    monkeypatch.setattr(S, '_infer', lambda *args: None)
    def scorer(argv, **kwargs):
        out = Path(argv[-1])
        out.mkdir()
        (out / 'scores.json').write_text(json.dumps(
            {'final_score': 0.2, 'macula_score': 0.4, 'widefield_score': 0.0}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(S.subprocess, 'run', scorer)
    report = S.score(a, b, 'published')
    assert report['status'] == 'MEASURED'
    assert report['reference_matches_exactly'] is False
    assert report['environment_matches_pin'] is False
    assert report['delta_from_published']['final_score'] == pytest.approx(-0.1)


def test_changed_official_scorer_warns_but_missing_scorer_errors(tmp_path, capsys):
    kit = tmp_path / 'starting_kit'
    directory = kit / 'app_scoring/program'
    directory.mkdir(parents=True)
    expected = {'needs': {'kit': {'files': {}}}}
    for name in ('scoring.py', 'metrics.py'):
        rel = 'starting_kit/app_scoring/program/' + name
        expected['needs']['kit']['files'][rel] = {'sha256': 'reference'}
        (directory / name).write_text('# local version\n')
    assert S._official_scorer(kit, expected) == directory / 'scoring.py'
    assert 'WARN' in capsys.readouterr().err
    (directory / 'metrics.py').unlink()
    with pytest.raises(ValueError, match='missing'):
        S._official_scorer(kit, expected)
