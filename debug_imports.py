print("디버깅 시작: 라이브러리 임포트 테스트...")

try:
    import numpy
    print("[성공] numpy 임포트 완료")
except Exception as e:
    print(f"[실패] numpy 임포트 중 오류: {e}")

try:
    import soundfile
    print("[성공] soundfile 임포트 완료")
except Exception as e:
    print(f"[실패] soundfile 임포트 중 오류: {e}")

try:
    import webrtcvad
    print("[성공] webrtcvad 임포트 완료")
except Exception as e:
    print(f"[실패] webrtcvad 임포트 중 오류: {e}")

try:
    import noisereduce
    print("[성공] noisereduce 임포트 완료")
except Exception as e:
    print(f"[실패] noisereduce 임포트 중 오류: {e}")

try:
    import pedalboard
    print("[성공] pedalboard 임포트 완료")
except Exception as e:
    print(f"[실패] pedalboard 임포트 중 오류: {e}")

print("\n모든 라이브러리 임포트 테스트 완료.")
