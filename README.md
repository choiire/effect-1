# Speech Volume Normalizer

잡음이 많은 음성 파일을 대상으로 한 견고한 음성 볼륨 평준화 도구입니다. VAD(Voice Activity Detection)와 DRC(Dynamic Range Compression)를 결합하여 음성 구간만을 선택적으로 압축하여 일관된 볼륨을 제공합니다.

## 주요 특징

- **견고한 음성 활동 감지**: WebRTC VAD를 사용하여 잡음 환경에서도 정확한 음성 구간 감지
- **선택적 압축**: 음성 구간에만 다이내믹 레인지 압축 적용
- **부드러운 전환**: 음성/비음성 구간 경계에서 크로스페이드 적용으로 아티팩트 방지
- **파라미터 조정 가능**: VAD 민감도, 압축 설정 등 세밀한 조정 가능
- **명령줄 인터페이스**: 사용하기 쉬운 CLI 제공

## 설치

1. 가상환경 생성 및 활성화 (권장):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용법
```bash
python main.py input.wav output_normalized.wav
```

### 분석 모드 (처리 없이 분석만)
```bash
python main.py input.wav output.wav --analyze-only
```

### 고급 옵션
```bash
python main.py input.wav output.wav \
    --threshold -18 \
    --ratio 3.0 \
    --attack 5 \
    --release 150 \
    --vad-mode 3 \
    --crossfade 15
```

## 파라미터 설명

### VAD (Voice Activity Detection) 파라미터
- `--vad-mode`: VAD 공격성 모드 (0-3, 높을수록 더 공격적)
- `--vad-frame-duration`: VAD 프레임 길이 (10, 20, 30ms)
- `--vad-hangover`: 음성 종료 후 유지 시간 (ms)

### DRC (Dynamic Range Compression) 파라미터
- `--threshold`: 압축 임계값 (dB, 예: -20)
- `--ratio`: 압축비 (예: 4.0은 4:1 압축)
- `--attack`: 어택 타임 (ms)
- `--release`: 릴리즈 타임 (ms)

### 처리 파라미터
- `--crossfade`: 구간 경계 크로스페이드 시간 (ms)
- `--block-size`: 오디오 처리 블록 크기

## 파일 구조

```
effect-1/
├── main.py                 # CLI 메인 인터페이스
├── speech_normalizer.py    # 메인 처리 파이프라인
├── vad_module.py          # VAD 모듈
├── drc_engine.py          # DRC 엔진
├── requirements.txt       # 의존성 목록
├── README.md             # 이 파일
└── input.wav             # 입력 예제 파일
```

## 알고리즘 개요

1. **음성 활동 감지 (VAD)**:
   - WebRTC VAD를 사용하여 음성 구간 식별
   - 16kHz 리샘플링 및 int16 변환
   - 행오버 로직으로 자연스러운 세그먼테이션

2. **다이내믹 레인지 압축 (DRC)**:
   - RMS 엔벨로프 팔로잉
   - 임계값 기반 게인 감소
   - 어택/릴리즈 타임 제어

3. **VAD-게이트 처리**:
   - 음성 구간에만 압축 적용
   - 비음성 구간은 원본 유지
   - 경계에서 크로스페이드 적용

## 예제

### 분석 예제
```bash
python main.py input.wav output.wav --analyze-only
```

출력 예시:
```
=== Analysis Results ===
Total duration: 10.50 seconds
Speech duration: 7.20 seconds (68.6%)
Number of speech segments: 5
Overall RMS level: 0.045231
Speech RMS level: 0.067845
```

### 처리 예제
```bash
python main.py input.wav normalized_output.wav --threshold -18 --ratio 3.0
```

## 팁

1. **잡음이 많은 환경**: `--vad-mode 3` 사용
2. **깨끗한 환경**: `--vad-mode 1` 사용
3. **부드러운 압축**: 낮은 ratio (2.0-3.0) 사용
4. **강한 압축**: 높은 ratio (4.0-8.0) 사용
5. **자연스러운 소리**: 긴 attack/release 시간 사용

## 문제 해결

- **음성이 감지되지 않음**: VAD 모드를 낮춰보세요 (`--vad-mode 0`)
- **너무 많은 잡음이 음성으로 감지됨**: VAD 모드를 높여보세요 (`--vad-mode 3`)
- **압축이 너무 강함**: threshold를 낮추거나 ratio를 줄여보세요
- **부자연스러운 소리**: crossfade 시간을 늘려보세요

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.
