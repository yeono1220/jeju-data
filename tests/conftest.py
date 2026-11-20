import pytest

from dataolle.analysis import Catalog
from dataolle.synth.generator import generate


@pytest.fixture(scope="session")
def tamna_cat(tmp_path_factory):
    """코스 4(탐나는전 캐시백 설계)·에이전트 시험용 가상데이터: 탐나는전 21개월(2024.1~2025.9)·카드 9개월·날씨 12개월 + 유동인구 v01·v02 각 1개월(하루치)."""
    out = tmp_path_factory.mktemp("tamna")
    generate(out, profile="tamna", pop_days=1, pop_hist_days=1, verbose=False)
    return Catalog.from_folder(out)
