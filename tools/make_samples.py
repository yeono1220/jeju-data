"""docs/samples(가상데이터 결과 예시 PDF)를 다시 만든다.

사용: python tools/make_samples.py <가상데이터 폴더(profile demo)>
가상데이터는 `python -m dataolle.synth.generator <폴더> --profile demo`로 만든다.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataolle.ai.agent import Agent  # noqa: E402
from dataolle.analysis import Catalog  # noqa: E402
from dataolle.courses.action_pack import make_action_pack  # noqa: E402
from dataolle.courses.before_after import BeforeAfterCourse, BeforeAfterInput  # noqa: E402
from dataolle.courses.market import MarketCourse, MarketInput  # noqa: E402
from dataolle.courses.policy import PolicyInput  # noqa: E402
from dataolle.courses.status import StatusCourse  # noqa: E402
from dataolle.courses.tamna import TamnaCourse, TamnaInput  # noqa: E402
from dataolle.report.export_pack import build_pack  # noqa: E402

OUT = ROOT / "docs" / "samples"
P = "[가상데이터] "


def save(src: Path | None, name: str) -> None:
    if src is None or not Path(src).exists():
        print("  (없음)", name)
        return
    dst = OUT / f"{P}{name}.pdf"
    shutil.copy2(src, dst)
    print(" ", dst.name)


def main(folder: str) -> None:
    cat = Catalog.from_folder(folder)
    work = Path(tempfile.mkdtemp(prefix="samples_"))
    for f in OUT.glob("*.pdf"):
        f.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1·2: 캐시백 설계(예산 400억 원, 1년) — 에이전트 규칙 경로 그대로
    res = Agent(cat, work / "a1", None).run("예산 400억 원으로 탐나는전 적립을 1년 내내 이어 가려면 어떻게 나눠야 해요?")
    o = res.outputs[-1]
    save(o.pdf, "1_탐나는전_캐시백설계_예산400억_제주전체")
    if res.packs:
        save(res.packs[-1].pdf, f"2_실행물묶음_제주전체_{res.packs[-1].scenario}안")

    # 3: 동네 하나(일도1동) 진단 + 쏠림 방지 우선
    o3 = TamnaCourse(cat, work / "t3").run(TamnaInput(scope="hd", value="일도1동", policy=PolicyInput(goal="fair")))
    save(o3.pdf, "3_탐나는전_동네진단과쏠림방지_일도1동")
    pk = make_action_pack(o3, work / "p3", "E")
    save(pk.pdf, "4_실행물묶음_일도1동_E안")

    # 5: 효과 확인 → 다음 배분(C′)
    o5 = TamnaCourse(cat, work / "t5").run(TamnaInput(event="202506", event_name="전통시장 추가 적립(가상 시나리오)"))
    save(o5.pdf, "5_탐나는전_효과확인과다음배분_제주전체")

    # 6~8: 같은 엔진의 보조 코스
    o6 = BeforeAfterCourse(cat, work / "b6").run(BeforeAfterInput(treat=["한림읍"], event="202507", biz="비알콜 음료점업",
                                                                  event_name="(가상 시나리오)"))
    save(o6.pdf, "6_사건전후효과_한림읍_카페")
    o7 = StatusCourse(cat, work / "s7").run_status("한림읍")
    save(o7.pdf, "7_동네현황자료_한림읍")
    o8 = MarketCourse(cat, work / "m8").run(MarketInput(biz="비알콜 음료점업", bjd="애월읍"))
    save(o8.pdf, "8_창업시장분석_애월읍_카페")

    # 9: 반출 신청 설명자료(일도1동 보고서 + 실행물)
    ep = build_pack(o3.report, work / "export", use="policy", report_pdf=o3.pdf,
                    extra_pdfs=[(pk.pdf, f"실행물 묶음({pk.scenario}안)")])
    desc = next((f.path for f in ep.files if f.role == "설명자료"), None)
    save(desc, "9_반출신청_설명자료_일도1동")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "요청파일"))
