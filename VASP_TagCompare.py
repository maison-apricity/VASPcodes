#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from collections import OrderedDict, defaultdict

# ============================================================
# 0) 완전히 무시할 키/정보
# ============================================================
IGNORE_KEYS = {
    "COPYR",
    "SHA256",
}

# ============================================================
# 1) 결과/iteration 로그 시작 라인 패턴
# ============================================================
SKIP_LINE_PATTERNS = [
    re.compile(r'^\s*Iteration', re.IGNORECASE),
    re.compile(r'^\s*DAV:', re.IGNORECASE),
    re.compile(r'^\s*RMM:', re.IGNORECASE),
    re.compile(r'^\s*CG\s*:', re.IGNORECASE),
    re.compile(r'^\s*N\s+E\s+dE', re.IGNORECASE),
    re.compile(r'^\s*POSITION\b', re.IGNORECASE),
    re.compile(r'^\s*TOTAL-FORCE', re.IGNORECASE),
    re.compile(r'^\s*FREE ENERGIE', re.IGNORECASE),
    re.compile(r'^\s*FORCES:', re.IGNORECASE),
    re.compile(r'^\s*E-fermi', re.IGNORECASE),
    re.compile(r'^\s*magnetization', re.IGNORECASE),
    re.compile(r'^\s*total charge', re.IGNORECASE),
    re.compile(r'^\s*soft charge-density', re.IGNORECASE),
    re.compile(r'^\s*average \(electrostatic\) potential', re.IGNORECASE),
    re.compile(r'^\s*LOOP:', re.IGNORECASE),
]

# ============================================================
# 2) 일반 KEY = VALUE 추출
# ============================================================
PAIR_RE = re.compile(
    r'([A-Z][A-Z0-9_()\/+\-]*)\s*=\s*(.*?)(?=(?:\s+[A-Z][A-Z0-9_()\/+\-]*\s*=)|$)'
)

NUM_RE = re.compile(
    r'(?<![\w./+\-*])([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[DdEe][+-]?\d+)?)(?![\w./+\-*])'
)

# ============================================================
# 3) 다단어 key / 특수 라인
# ============================================================
SPECIAL_PATTERNS = [
    (re.compile(r'^\s*ions per type\s*=\s*(.+)$', re.IGNORECASE), 'IONS_PER_TYPE'),
    (re.compile(r'^\s*generate k-points for:\s*(.+)$', re.IGNORECASE), 'KPOINTS_GRID'),
    (re.compile(r'^\s*shift:\s*(.+)$', re.IGNORECASE), 'KPOINTS_SHIFT'),
    (re.compile(r'^\s*dimension of arrays:\s*(.+)$', re.IGNORECASE), 'DIMENSION_OF_ARRAYS'),
]

# ============================================================
# 4) 카테고리 분류용 키 세트
# ============================================================
ACTUAL_SETTING_KEYS = {
    # 기본/정확도/전자
    "PREC", "ENCUT", "ENINI", "EDIFF", "EDIFFG", "NELM", "NELMIN", "NELMDL",
    "ALGO", "IALGO", "LREAL", "ADDGRID", "LASPH", "LMAXMIX", "LMAXPAW",
    "ISMEAR", "SIGMA", "AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "MAXMIX",
    "WEIMIN", "NELECT", "NBANDS", "NCORE", "NPAR", "KPAR", "NSIM",
    "ISPIN", "MAGMOM", "SAXIS",
    # 구조 최적화/NEB
    "IBRION", "NSW", "POTIM", "ISIF", "ISYM",
    "SPRING", "LCLIMB", "IMAGES", "ICHAIN", "IOPT", "LTANGENTOLD",
    # dipole / electrostatics
    "LDIPOL", "IDIPOL", "DIPOL", "LMONO", "EFIELD", "EFIELD_PEAD",
    # vdW / XC / hybrid
    "IVDW", "GGA", "METAGGA", "LUSE_VDW", "ZAB_VDW", "PARAM1", "PARAM2",
    "LHFCALC", "AEXX", "HFSCREEN", "TIME", "PRECFOCK",
    # 격자/그리드
    "NGX", "NGY", "NGZ", "NGXF", "NGYF", "NGZF",
    "ENAUG",
    # 기타 계산 모델
    "LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT",
    "LSORBIT", "LNONCOLLINEAR", "GGA_COMPAT", "LASYNC",
}

RESTART_OUTPUT_KEYS = {
    "ISTART", "ICHARG", "INIWAV",
    "LWAVE", "LCHARG", "LVHAR", "LVTOT", "LAECHG", "LELF",
    "LOPTICS", "LEPSILON", "LCALCEPS", "NWRITE", "LORBIT", "LPARD",
    "LSEPB", "LSEPK", "LPLANE", "NEDOS",
}

POTCAR_SYSTEM_KEYS = {
    # POTCAR 관련
    "TITEL", "VRHFIN", "ZVAL", "POMASS", "ENMAX", "ENMIN",
    "RCORE", "RAUG", "RWIGS", "RMAX", "RDEP", "RDEPT", "RPACOR",
    # 시스템/구조/격자/점 개수
    "SYSTEM", "NIONS", "NTYP", "IONS_PER_TYPE", "NKPTS", "KPOINTS_GRID",
    "KPOINTS_SHIFT", "DIMENSION_OF_ARRAYS",
    "LATTICE_A", "LATTICE_B", "LATTICE_C", "CELL_VOLUME",
}

RESULT_HISTORY_KEYS = {
    "TOTEN", "E0", "F", "E", "DENC", "EBANDS", "EENTRO", "PSCENC",
    "TEWEN", "XCENC", "PAWDC1", "PAWDC2", "GAMMA", "EKIN", "ETOTAL",
    "EAUG", "EATOM",
}

# 여러 토큰을 전부 유지해야 하는 키
FULL_VALUE_KEYS = {
    "DIPOL", "MAGMOM", "SAXIS",
    "LDAUL", "LDAUU", "LDAUJ",
    "IONS_PER_TYPE", "KPOINTS_GRID", "KPOINTS_SHIFT",
    "DIMENSION_OF_ARRAYS",
    "TITEL", "VRHFIN",
    "LATTICE_A", "LATTICE_B", "LATTICE_C",
}

BOOL_TRUE = {"TRUE", ".TRUE.", "T"}
BOOL_FALSE = {"FALSE", ".FALSE.", "F"}


def should_skip_line(line: str) -> bool:
    return any(p.search(line) for p in SKIP_LINE_PATTERNS)


def normalize_key(key: str) -> str:
    key = key.strip().upper()
    key = re.sub(r'\s+', '_', key)
    return key


def strip_annotations(text: str) -> str:
    # # 이후 제거
    text = re.sub(r'#.*$', '', text)

    # 괄호 안 반복 제거
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'\([^()]*\)', '', text)

    return text.strip()


def normalize_numeric_tokens(text: str) -> str:
    def repl(match):
        s = match.group(1).replace('D', 'E').replace('d', 'e')
        try:
            x = float(s)
            return f"{x:.12g}"
        except ValueError:
            return match.group(1)
    return NUM_RE.sub(repl, text)


def normalize_raw_value(value: str) -> str:
    value = strip_annotations(value)
    value = value.strip().strip(';').strip(',').strip()
    value = value.upper()

    for t in BOOL_TRUE:
        value = value.replace(t, "TRUE")
    for t in BOOL_FALSE:
        value = value.replace(t, "FALSE")

    value = normalize_numeric_tokens(value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def is_numeric_token(tok: str) -> bool:
    return re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?', tok) is not None


def core_value(key: str, raw_value: str) -> str:
    """
    비교용 핵심값.
    - FULL_VALUE_KEYS는 전체 보존
    - 일반 키는 첫 토큰만
    """
    if key in FULL_VALUE_KEYS:
        return raw_value

    toks = raw_value.split()
    if not toks:
        return raw_value

    # 일반적으로 첫 토큰만 비교
    return toks[0].strip(',').strip()


def add_entry(store: OrderedDict, key: str, raw_value: str):
    if key in IGNORE_KEYS:
        return

    raw_value = normalize_raw_value(raw_value)
    if not raw_value:
        return

    cval = core_value(key, raw_value)

    if key not in store:
        store[key] = {
            "raw_values": [],
            "core_values": [],
            "last_raw": None,
            "last_core": None,
        }

    if raw_value not in store[key]["raw_values"]:
        store[key]["raw_values"].append(raw_value)
    if cval not in store[key]["core_values"]:
        store[key]["core_values"].append(cval)

    store[key]["last_raw"] = raw_value
    store[key]["last_core"] = cval


def classify_key(key: str) -> str:
    if key in ACTUAL_SETTING_KEYS:
        return "actual"
    if key in RESTART_OUTPUT_KEYS:
        return "restart_output"
    if key in POTCAR_SYSTEM_KEYS:
        return "potcar_system"
    if key in RESULT_HISTORY_KEYS:
        return "result_history"
    return "unknown"


def parse_lattice_block(lines, idx, store):
    """
    OUTCAR의 'direct lattice vectors' 블록 파싱
    """
    header = lines[idx]
    if "direct lattice vectors" not in header.lower():
        return

    # 다음 3줄 기대
    for j, name in enumerate(["LATTICE_A", "LATTICE_B", "LATTICE_C"], start=1):
        if idx + j >= len(lines):
            return
        parts = lines[idx + j].split()
        if len(parts) < 3:
            return
        try:
            vec = " ".join(f"{float(parts[k]):.12g}" for k in range(3))
            add_entry(store, name, vec)
        except ValueError:
            return


def parse_cell_volume(lines, store):
    """
    OUTCAR에서 volume of cell 추출
    """
    vol_re = re.compile(r'volume of cell\s*:\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?)', re.IGNORECASE)
    for line in lines:
        m = vol_re.search(line)
        if m:
            add_entry(store, "CELL_VOLUME", m.group(1))
            break


def parse_outcar(path: str) -> OrderedDict:
    data = OrderedDict()

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)

    # lattice / volume 선파싱
    for i, line in enumerate(lines):
        if "direct lattice vectors" in line.lower():
            parse_lattice_block(lines, i, data)
            break
    parse_cell_volume(lines, data)

    # 일반 라인 파싱
    for line in lines:
        if should_skip_line(line):
            continue

        # 특수 다단어 key
        special_hit = False
        for pattern, skey in SPECIAL_PATTERNS:
            m = pattern.search(line)
            if m:
                add_entry(data, normalize_key(skey), m.group(1))
                special_hit = True
                break
        if special_hit:
            continue

        # 일반 KEY=VALUE
        matches = PAIR_RE.findall(line)
        if matches:
            for raw_key, raw_val in matches:
                key = normalize_key(raw_key)
                add_entry(data, key, raw_val)

    return data


def summarize_history(info: dict) -> str:
    nuniq = len(info["core_values"])
    lastv = info["last_core"] if info["last_core"] is not None else "-"
    return f"uniq={nuniq}, last={lastv}"


def compare_category(d1: OrderedDict, d2: OrderedDict, category: str):
    diff_rows = []
    only1_rows = []
    only2_rows = []

    keys = sorted(set(d1.keys()) | set(d2.keys()))
    for key in keys:
        if classify_key(key) != category:
            continue

        in1 = key in d1
        in2 = key in d2

        if in1 and in2:
            v1 = d1[key]["core_values"]
            v2 = d2[key]["core_values"]

            # POTCAR/시스템 정보는 full sequence 비교
            if category == "potcar_system":
                s1 = " || ".join(v1)
                s2 = " || ".join(v2)
            elif category == "result_history":
                s1 = summarize_history(d1[key])
                s2 = summarize_history(d2[key])
            else:
                s1 = d1[key]["last_core"] if d1[key]["last_core"] is not None else "-"
                s2 = d2[key]["last_core"] if d2[key]["last_core"] is not None else "-"

            if s1 != s2:
                diff_rows.append((key, s1, s2))

        elif in1 and not in2:
            if category == "potcar_system":
                s1 = " || ".join(d1[key]["core_values"])
            elif category == "result_history":
                s1 = summarize_history(d1[key])
            else:
                s1 = d1[key]["last_core"] if d1[key]["last_core"] is not None else "-"
            only1_rows.append((key, s1))

        elif in2 and not in1:
            if category == "potcar_system":
                s2 = " || ".join(d2[key]["core_values"])
            elif category == "result_history":
                s2 = summarize_history(d2[key])
            else:
                s2 = d2[key]["last_core"] if d2[key]["last_core"] is not None else "-"
            only2_rows.append((key, s2))

    return diff_rows, only1_rows, only2_rows


def compare_unknown_stable(d1: OrderedDict, d2: OrderedDict):
    """
    unknown 중에서도 stable한 항목은 별도 섹션으로 보여줌
    """
    diff_rows = []
    only1_rows = []
    only2_rows = []

    keys = sorted(set(d1.keys()) | set(d2.keys()))
    for key in keys:
        if classify_key(key) != "unknown":
            continue

        in1 = key in d1
        in2 = key in d2

        def is_stable(info):
            return len(info["core_values"]) == 1

        if in1 and in2:
            if is_stable(d1[key]) and is_stable(d2[key]):
                s1 = d1[key]["last_core"]
                s2 = d2[key]["last_core"]
                if s1 != s2:
                    diff_rows.append((key, s1, s2))
        elif in1 and not in2:
            if is_stable(d1[key]):
                only1_rows.append((key, d1[key]["last_core"]))
        elif in2 and not in1:
            if is_stable(d2[key]):
                only2_rows.append((key, d2[key]["last_core"]))

    return diff_rows, only1_rows, only2_rows


def compare_unknown_dynamic(d1: OrderedDict, d2: OrderedDict):
    """
    unknown 중 dynamic한 항목은 결과/이력 참고용으로 요약
    """
    diff_rows = []
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    for key in keys:
        if classify_key(key) != "unknown":
            continue

        in1 = key in d1
        in2 = key in d2

        def is_dynamic(info):
            return len(info["core_values"]) > 1

        if in1 and in2 and (is_dynamic(d1[key]) or is_dynamic(d2[key])):
            s1 = summarize_history(d1[key])
            s2 = summarize_history(d2[key])
            if s1 != s2:
                diff_rows.append((key, s1, s2))
        elif in1 and not in2 and is_dynamic(d1[key]):
            diff_rows.append((key, summarize_history(d1[key]), "-"))
        elif in2 and not in1 and is_dynamic(d2[key]):
            diff_rows.append((key, "-", summarize_history(d2[key])))

    return diff_rows


def print_diff_table(title, rows, key_w):
    print(f"\n[{title}]")
    if not rows:
        print("(없음)")
        return

    print(f"{'KEY':<{key_w}} | OUTCAR1 | OUTCAR2")
    print(f"{'-'*key_w}-+-{'-'*40}-+-{'-'*40}")
    for k, a, b in rows:
        print(f"{k:<{key_w}} | {a} | {b}")


def print_single_table(title, rows, key_w):
    print(f"\n[{title}]")
    if not rows:
        print("(없음)")
        return

    print(f"{'KEY':<{key_w}} | VALUE")
    print(f"{'-'*key_w}-+-{'-'*40}")
    for k, a in rows:
        print(f"{k:<{key_w}} | {a}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "두 개의 VASP OUTCAR에서 계산 결과/iteration 로그를 제외하고, "
            "실제 계산 설정 / 재시작·출력 설정 / POTCAR·시스템 메타데이터 / 결과·이력 항목으로 나누어 비교합니다."
        )
    )
    parser.add_argument("outcar1", help="첫 번째 OUTCAR")
    parser.add_argument("outcar2", help="두 번째 OUTCAR")
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="결과/이력 항목 및 unknown dynamic 항목 요약도 출력"
    )
    parser.add_argument(
        "--show-count",
        action="store_true",
        help="수집된 키 개수 출력"
    )
    args = parser.parse_args()

    d1 = parse_outcar(args.outcar1)
    d2 = parse_outcar(args.outcar2)

    if args.show_count:
        print(f"[정보] OUTCAR1 수집 키 수: {len(d1)}")
        print(f"[정보] OUTCAR2 수집 키 수: {len(d2)}")

    key_w = max([24] + [len(k) for k in set(list(d1.keys()) + list(d2.keys()))])

    # 1) 실제 계산 설정
    diff_a, only1_a, only2_a = compare_category(d1, d2, "actual")
    print_diff_table("실제 계산 설정 차이", diff_a, key_w)
    print_single_table("OUTCAR1에만 있는 실제 계산 설정", only1_a, key_w)
    print_single_table("OUTCAR2에만 있는 실제 계산 설정", only2_a, key_w)

    # 2) 재시작/출력 설정
    diff_r, only1_r, only2_r = compare_category(d1, d2, "restart_output")
    print_diff_table("재시작/출력 설정 차이", diff_r, key_w)
    print_single_table("OUTCAR1에만 있는 재시작/출력 설정", only1_r, key_w)
    print_single_table("OUTCAR2에만 있는 재시작/출력 설정", only2_r, key_w)

    # 3) POTCAR/시스템 메타데이터
    diff_p, only1_p, only2_p = compare_category(d1, d2, "potcar_system")
    print_diff_table("POTCAR/시스템 메타데이터 차이", diff_p, key_w)
    print_single_table("OUTCAR1에만 있는 POTCAR/시스템 메타데이터", only1_p, key_w)
    print_single_table("OUTCAR2에만 있는 POTCAR/시스템 메타데이터", only2_p, key_w)

    # 4) unknown stable
    diff_u, only1_u, only2_u = compare_unknown_stable(d1, d2)
    print_diff_table("기타 안정 항목 차이(분류 미지정)", diff_u, key_w)
    print_single_table("OUTCAR1에만 있는 기타 안정 항목", only1_u, key_w)
    print_single_table("OUTCAR2에만 있는 기타 안정 항목", only2_u, key_w)

    # 5) 결과/이력 항목은 요청 시만
    if args.show_history:
        diff_h, only1_h, only2_h = compare_category(d1, d2, "result_history")
        print_diff_table("결과/이력 항목 요약(참고용)", diff_h, key_w)
        print_single_table("OUTCAR1에만 있는 결과/이력 항목", only1_h, key_w)
        print_single_table("OUTCAR2에만 있는 결과/이력 항목", only2_h, key_w)

        diff_ud = compare_unknown_dynamic(d1, d2)
        print_diff_table("기타 동적 항목 요약(분류 미지정)", diff_ud, key_w)


if __name__ == "__main__":
    main()