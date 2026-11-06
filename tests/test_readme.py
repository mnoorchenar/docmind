"""GitHub and Hugging Face show different files, which must keep the same content."""

import re

from scripts import sync_readme


def test_root_readme_matches_the_github_readme():
    actual = sync_readme.TARGET.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert actual == sync_readme.build(), "Run: python -m scripts.sync_readme"


def test_root_readme_keeps_the_hugging_face_settings():
    settings = sync_readme.build().split("---")[1]
    assert "sdk: docker" in settings
    assert "app_port: 7860" in settings


def test_local_files_in_both_readmes_exist():
    source = sync_readme.SOURCE.read_text(encoding="utf-8")
    paths = re.findall(r'(?:src="|\]\()\.\./([^")#]+)', source)
    assert paths
    for path in paths:
        assert (sync_readme.ROOT / path).exists(), path
