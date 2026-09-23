import pytest

from byteff2.utils import ASSET_ROOT_ENV, get_asset_path


def test_get_asset_path_prefers_explicit_root(monkeypatch, tmp_path):
    env_root = tmp_path / "env"
    explicit_root = tmp_path / "explicit"
    (env_root / "release").mkdir(parents=True)
    expected = explicit_root / "release/model"
    expected.mkdir(parents=True)
    monkeypatch.setenv(ASSET_ROOT_ENV, str(env_root))

    assert get_asset_path("release/model", explicit_root) == str(expected.resolve())


def test_get_asset_path_uses_environment_from_any_cwd(monkeypatch, tmp_path):
    asset_root = tmp_path / "assets"
    expected = asset_root / "release/data.txt"
    expected.parent.mkdir(parents=True)
    expected.write_text("data", encoding="utf-8")
    other_cwd = tmp_path / "work"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    monkeypatch.setenv(ASSET_ROOT_ENV, str(asset_root))

    assert get_asset_path("release/data.txt") == str(expected.resolve())


def test_get_asset_path_reports_configuration_and_missing_assets(monkeypatch, tmp_path):
    monkeypatch.delenv(ASSET_ROOT_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=ASSET_ROOT_ENV):
        get_asset_path("release/model")
    with pytest.raises(FileNotFoundError, match="release/model"):
        get_asset_path("release/model", tmp_path)


@pytest.mark.parametrize("relative_path", ["/absolute", "../outside", "release/../../outside"])
def test_get_asset_path_rejects_paths_outside_root(tmp_path, relative_path):
    with pytest.raises(ValueError, match="below the asset root"):
        get_asset_path(relative_path, tmp_path)
