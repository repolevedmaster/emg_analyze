import pickle
import numpy as np
import sys

# Windows 콘솔 유니코드 출력(cp949 에러) 방지
sys.stdout.reconfigure(encoding='utf-8')

DATA_FILE = "data.pickle"

def inspect_pickle(file_path):
    print(f"[{file_path}] 파일 분석을 시작합니다...\n")
    
    try:
        with open(file_path, "rb") as f:
            raw = pickle.load(f)
            
        print("1. 파일 내 최상위 키(Keys):", list(raw.keys()))
        
        if "data" in raw and "labels" in raw:
            data = raw["data"]
            labels = raw["labels"]
            
            print(f"\n2. 총 데이터 개수: {len(data)} 개")
            
            # 클래스(라벨) 분포 확인
            unique_labels = list(set(labels))
            print(f"3. 고유 라벨(문장) 수: {len(unique_labels)} 개")
            print("   [라벨 목록]:")
            for lbl in unique_labels:
                print(f"    - {lbl}")
                
            # 데이터 구조 및 shape 확인
            print("\n4. 데이터 Shape 분석 (첫 3개 샘플):")
            for i in range(min(3, len(data))):
                sample = data[i]
                if isinstance(sample, np.ndarray):
                    shape_info = sample.shape
                    dtype_info = sample.dtype
                elif isinstance(sample, list):
                    # 리스트인 경우 numpy 배열로 임시 변환하여 차원 확인
                    arr = np.array(sample)
                    shape_info = arr.shape
                    dtype_info = arr.dtype
                else:
                    shape_info = "Unknown (not array or list)"
                    dtype_info = type(sample)
                    
                print(f"   - 샘플 {i+1} : shape={shape_info}, type={dtype_info}")
                
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    inspect_pickle(DATA_FILE)
