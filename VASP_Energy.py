import py4vasp
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로
file_path = r"C:\Users\dlstj\Desktop\Choi\vaspout.h5"

# 데이터 로딩
data = py4vasp.Calculation.from_file(file_path)

# 1. 에너지 추출
energy_raw = data.energy

# 2. to_dict(): 단일 snapshot
try:
    energy_dict = energy_raw.to_dict()
    print("\n▶ to_dict() 출력:")
    for k, v in energy_dict.items():
        print(f"{k}: {v}")
except Exception as e:
    print("to_dict() 실패:", e)

# 3. to_numpy(): 전체 step 확인
try:
    energy_np = energy_raw.to_numpy()
    print("\n▶ to_numpy() 출력:")
    print("type:", type(energy_np))
    print("value:", energy_np)
    print("shape:", getattr(energy_np, 'shape', 'shape 없음'))

    # 경우 분기
    if isinstance(energy_np, np.ndarray):
        if energy_np.ndim == 2:
            print("✅ 2차원 배열 → step별 에너지 존재")
            print("총 step 수:", energy_np.shape[0])
            print("에너지 종류 수:", energy_np.shape[1])
        elif energy_np.ndim == 1:
            print("⚠ 단일 ionic step만 포함됨 (1차원)")
        elif energy_np.ndim == 0:
            print("⚠ 단일 에너지 값만 포함됨 (스칼라)")
    else:
        print("⚠ 반환값이 배열이 아님:", energy_np)

except Exception as e:
    print("to_numpy() 실패:", e)
