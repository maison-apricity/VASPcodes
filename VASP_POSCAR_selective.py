#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# =========================
# 사용자 설정
# =========================
INPUT_POSCAR = r"./POSCAR"
OUTPUT_POSCAR = r"./POSCAR_modified"

# zero-based atom indices
TARGET_INDICES = {
    176, 181, 182, 183
}


def is_integer_line(tokens):
    try:
        for t in tokens:
            int(t)
        return True
    except ValueError:
        return False


def main():
    lines = Path(INPUT_POSCAR).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError("POSCAR 형식이 너무 짧습니다.")

    # POSCAR header
    comment = lines[0]
    scale = lines[1]
    lattice = lines[2:5]

    idx = 5

    # VASP 5 여부 판별
    tokens = lines[idx].split()
    if is_integer_line(tokens):
        # VASP 4 style: counts only
        species_line = None
        counts_line = lines[idx]
        counts = [int(x) for x in tokens]
        idx += 1
    else:
        # VASP 5 style: species then counts
        species_line = lines[idx]
        idx += 1
        counts_line = lines[idx]
        counts = [int(x) for x in counts_line.split()]
        idx += 1

    natoms = sum(counts)

    # Selective dynamics 여부 확인
    selective_present = False
    if idx < len(lines) and lines[idx].strip().lower().startswith("s"):
        selective_present = True
        idx += 1

    # Coordinate mode
    if idx >= len(lines):
        raise ValueError("좌표 모드 줄(Direct/Cartesian)을 찾을 수 없습니다.")
    coord_mode = lines[idx]
    idx += 1

    coord_mode_lower = coord_mode.strip().lower()
    if not (coord_mode_lower.startswith("d") or coord_mode_lower.startswith("c") or coord_mode_lower.startswith("k")):
        raise ValueError(f"좌표 모드가 이상합니다: {coord_mode}")

    # Coordinate lines
    coord_lines = lines[idx:idx + natoms]
    if len(coord_lines) != natoms:
        raise ValueError(f"원자 수({natoms})와 좌표 줄 수({len(coord_lines)})가 맞지 않습니다.")

    max_index = natoms - 1
    bad = sorted(i for i in TARGET_INDICES if i < 0 or i > max_index)
    if bad:
        raise IndexError(
            f"유효 범위를 벗어난 zero-based 인덱스가 있습니다. "
            f"허용 범위: 0 ~ {max_index}, 문제 인덱스: {bad[:20]}"
            + (" ..." if len(bad) > 20 else "")
        )

    new_coord_lines = []
    for i, line in enumerate(coord_lines):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{i}번 원자의 좌표 줄 형식이 잘못되었습니다: {line}")

        xyz = parts[:3]

        # 기존 SD가 있으면 기존 flags는 버리고, 뒤에 남는 주석류만 보존
        if selective_present:
            tail = parts[6:] if len(parts) >= 6 else []
        else:
            tail = parts[3:] if len(parts) > 3 else []

        flags = ["T", "T", "T"] if i in TARGET_INDICES else ["F", "F", "F"]

        rebuilt = xyz + flags + tail
        new_coord_lines.append(" ".join(rebuilt))

    # Output reconstruction
    out = []
    out.append(comment)
    out.append(scale)
    out.extend(lattice)
    if species_line is not None:
        out.append(species_line)
    out.append(counts_line)
    out.append("Selective dynamics")
    out.append(coord_mode)
    out.extend(new_coord_lines)

    Path(OUTPUT_POSCAR).write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"[완료] 입력 파일 : {INPUT_POSCAR}")
    print(f"[완료] 출력 파일 : {OUTPUT_POSCAR}")
    print(f"[정보] 전체 원자 수 : {natoms}")
    print(f"[정보] T T T 원자 수 : {len(TARGET_INDICES)}")
    print(f"[정보] F F F 원자 수 : {natoms - len(TARGET_INDICES)}")
    print(f"[정보] zero-based 최대 인덱스 : {max_index}")


if __name__ == "__main__":
    main()