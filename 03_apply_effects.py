import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, Limiter, HighpassFilter, PeakFilter
import argparse

def apply_effects(input_path, output_path):
    print(f"\n--- 3단계: 오디오 효과 적용 및 최종 출력 ---")
    print(f"입력 파일: {input_path}")

    try:
        audio_data, sample_rate = sf.read(input_path, dtype='float32')
    except Exception as e:
        print(f"오류: 오디오 파일을 읽을 수 없습니다. {e}")
        return

    if audio_data.size == 0:
        print("입력 오디오가 비어있어 처리를 건너뜁니다.")
        sf.write(output_path, np.array([], dtype=np.float32), sample_rate)
        return

    print("오디오 효과를 적용합니다: 하이패스, 컴프레서, EQ, 리미터...")
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-16, ratio=2.5, attack_ms=5, release_ms=100),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=3, q=1.0), # 목소리 선명도 향상
        Limiter(threshold_db=-1.0, release_ms=50)
    ])

    processed_audio = board(audio_data, sample_rate)

    try:
        sf.write(output_path, processed_audio, sample_rate)
        print(f"모든 처리 완료. 최종 결과 저장: {output_path}")
    except Exception as e:
        print(f"오류: 최종 파일을 저장할 수 없습니다. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="오디오 파일에 효과를 적용합니다.")
    parser.add_argument("-i", "--input", required=True, help="입력 오디오 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 오디오 파일 경로")
    args = parser.parse_args()

    apply_effects(args.input, args.output)
