"""행정안전부 주민등록 인구통계 '연령별 인구현황'(행정동, 5세 단위) 엑셀 → src/dataolle/spec/data/public_pop.json

사용: python tools/build_public_pop.py <내려받은 xlsx> [<xlsx> ...]
받는 곳: https://jumin.mois.go.kr → 연령별 인구현황 → 제주특별자치도 제주시 / 서귀포시, 5세 단위, 최근 달.
원본 엑셀은 저장소·ZIP에 넣지 않고, 읍면동별 총인구·65세 이상·0~14세만 뽑아 JSON으로 둔다(공개 통계, 출처 표시).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "dataolle" / "spec" / "data" / "public_pop.json"
OLD = ("65~69세", "70~74세", "75~79세", "80~84세", "85~89세", "90~94세", "95~99세", "100세 이상")
CHILD = ("0~4세", "5~9세", "10~14세")


def num(x) -> int:
    return int(str(x).replace(",", "").strip() or 0)


def read(path: str) -> tuple[str, list[dict]]:
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    ym = next((re.sub(r"\D", "", str(c)) for r in rows[:4] for c in r if c and re.fullmatch(r"\d{4}년\s*\d{1,2}월", str(c).strip())), "")
    ym = ym[:4] + ym[4:].zfill(2) if ym else ""
    hi = next(i for i, r in enumerate(rows) if r and str(r[0]).strip() == "행정기관코드")
    head = [str(c).strip() if c is not None else "" for c in rows[hi]]
    first = {}                                                     # '계' 묶음(맨 앞)의 열만 쓴다
    for j, h in enumerate(head):
        first.setdefault(h, j)
    out = []
    for r in rows[hi + 1:]:
        if not r or not r[0]:
            continue
        code, name = str(r[0]).strip(), str(r[1]).strip()
        parts = name.split()
        if len(parts) < 3 or parts[0] != "제주특별자치도":           # 시 합계 줄, 다른 시도(전국 파일을 받아도 제주만 남김)
            continue
        out.append({"code": code, "si": parts[1], "name": parts[-1], "total": num(r[first["총 거주자수"]]) if "총 거주자수" in first
                    else num(r[2]), "age65": sum(num(r[first[c]]) for c in OLD), "child014": sum(num(r[first[c]]) for c in CHILD)})
    return ym, out


def main(paths: list[str]) -> None:
    units: dict[str, dict] = {}
    yms = set()
    for p in paths:
        ym, rows = read(p)
        yms.add(ym)
        for r in rows:
            units[r["code"]] = r
    doc = {"title": "주민등록 인구통계 연령별 인구현황(행정동, 5세 단위)", "provider": "행정안전부", "url": "https://jumin.mois.go.kr",
           "ym": sorted(yms)[-1] if yms else "", "note": "총 거주자수·65세 이상·0~14세만 발췌. 매월 말일 기준 공표 통계",
           "units": sorted(units.values(), key=lambda r: r["code"])}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    si = {}
    for r in doc["units"]:
        si[r["si"]] = si.get(r["si"], 0) + 1
    print(f"{OUT.name}: {len(doc['units'])}개 읍면동({si}), 기준 {doc['ym']}, 총 {sum(r['total'] for r in doc['units']):,}명, "
          f"65세 이상 {sum(r['age65'] for r in doc['units']):,}명")


if __name__ == "__main__":
    main(sys.argv[1:])
