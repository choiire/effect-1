import soundfile as sf
import numpy as np
import noisereduce as nr
import argparse

def denoise_audio(input_path, output_path):
    print(f"--- 1단계: 노이즈 제거 시작 ---")
    print(f"입력 파일: {input_path}")

    try:
        audio_data, sample_rate = sf.read(input_path)
    except Exception as e:
        print(f"오류: 오디오 파일을 읽을 수 없습니다. {e}")
        return

    # 모노 채널로 변환
    if audio_data.ndim > 1:
        print("모노 채널로 변환합니다.")
        audio_data = np.mean(audio_data, axis=1).astype(np.float32)
    else:
        audio_data = audio_data.astype(np.float32)

    # 첫 1초를 노이즈로 가정하여 노이즈 제거
    print("전체 오디오 파일에 대해 노이즈를 제거합니다...")
    noise_clip = audio_data[:sample_rate]
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, y_noise=noise_clip, stationary=True)

    try:
        sf.write(output_path, reduced_noise_audio, sample_rate)
        print(f"노이즈 제거 완료. 결과 저장: {output_path}")
    except Exception as e:
        print(f"오류: 처리된 파일을 저장할 수 없습니다. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="오디오 파일의 노이즈를 제거합니다.")
    parser.add_argument("-i", "--input", required=True, help="입력 오디오 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 오디오 파일 경로")
    args = parser.parse_args()

    denoise_audio(args.input, args.output)
