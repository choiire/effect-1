import soundfile as sf
import numpy as np
import webrtcvad
import argparse
from collections import deque

# --- 상수 및 헬퍼 클래스/함수 (VAD 처리에 필요) ---
SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]
FRAME_DURATION_MS = 30

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, s in ring_buffer if s])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    if s:
                        voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, s in ring_buffer if not s])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield voiced_frames[0].timestamp, \
                      voiced_frames[-1].timestamp + voiced_frames[-1].duration
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield voiced_frames[0].timestamp, \
              voiced_frames[-1].timestamp + voiced_frames[-1].duration

def extract_speech(input_path, output_path):
    print(f"\n--- 2단계: 음성 구간 추출 시작 ---")
    print(f"입력 파일: {input_path}")

    try:
        audio_data, sample_rate = sf.read(input_path, dtype='float32')
    except Exception as e:
        print(f"오류: 오디오 파일을 읽을 수 없습니다. {e}")
        return

    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        print(f"지원되지 않는 샘플링 레이트입니다: {sample_rate}Hz.")
        return

    print("음성 활동을 감지하여 음성 구간을 찾습니다...")
    vad = webrtcvad.Vad(3)  # 가장 공격적인 레벨
    audio_int16 = (audio_data * 32767).astype(np.int16)
    frames = list(frame_generator(FRAME_DURATION_MS, audio_int16.tobytes(), sample_rate))
    segments = list(vad_collector(sample_rate, FRAME_DURATION_MS, 300, vad, frames))

    if not segments:
        print("음성 구간을 찾지 못했습니다. 빈 파일을 생성합니다.")
        sf.write(output_path, np.array([], dtype=np.float32), sample_rate)
        return

    print(f"{len(segments)}개의 음성 구간을 찾았습니다. 구간을 병합합니다...")
    speech_chunks = []
    for start_time, end_time in segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        speech_chunks.append(audio_data[start_sample:end_sample])
    
    speech_only_audio = np.concatenate(speech_chunks)

    try:
        sf.write(output_path, speech_only_audio, sample_rate)
        print(f"음성 구간 추출 완료. 결과 저장: {output_path}")
    except Exception as e:
        print(f"오류: 처리된 파일을 저장할 수 없습니다. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="오디오 파일에서 음성 구간만 추출합니다.")
    parser.add_argument("-i", "--input", required=True, help="입력 오디오 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 오디오 파일 경로")
    args = parser.parse_args()

    extract_speech(args.input, args.output)
