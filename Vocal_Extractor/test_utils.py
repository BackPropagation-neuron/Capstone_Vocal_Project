import json
import math
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

# 앞서 만든 코드가 'vocal_extractor.py'에 있다고 가정합니다.
from vocal_extractor import VocalFeatureExtractor

def generate_dummy_vocal(filename="dummy_vocal.wav", sr=22050, duration=3.0):
    """
    테스트를 위해 비브라토(떨림)가 포함된 440Hz(A4) 가상 오디오 파일을 생성합니다.
    """
    print(f"[{filename}] 테스트용 가상 보컬 오디오를 생성 중입니다...")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 기본 주파수 440Hz에 6Hz의 규칙적인 비브라토(떨림)를 추가
    f0 = 440
    vibrato_rate = 6.0  # 6Hz의 바이브레이션
    vibrato_extent = 15 # 주파수 변화폭
    
    # 시간에 따른 주파수 변화 배열 생성
    instantaneous_frequency = f0 + vibrato_extent * np.sin(2 * np.pi * vibrato_rate * t)
    
    # 주파수를 적분하여 위상(Phase) 계산 후 사인파 생성
    phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / sr
    audio = np.sin(phase)
    
    # 약간의 배음(Harmonics)과 화이트 노이즈를 섞어 실제 목소리와 비슷하게 만듦
    audio += 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    audio += np.random.normal(0, 0.01, audio.shape)
    
    # 16-bit PCM WAV 파일로 저장
    audio_normalized = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wavfile.write(filename, sr, audio_normalized)
    print("-> 오디오 생성 완료!\n")
    return filename

def plot_contour(f0_contour, rms_contour, sr=22050):
    """추출된 시계열 데이터를 시각화합니다."""
    print("[그래프] 추출된 시계열 데이터(F0, RMS)를 시각화합니다. (창을 닫으면 프로그램이 종료됩니다)")
    
    # pYIN의 기본 hop_length는 512입니다. 이를 기반으로 시간축 생성
    hop_length = 512
    times = np.arange(len(f0_contour)) * hop_length / sr

    plt.figure(figsize=(12, 6))
    
    # 1. Pitch (F0) 그래프
    plt.subplot(2, 1, 1)
    # 0(무음)인 부분은 그래프에서 끊어지게 보이도록 NaN 처리
    f0_plot = np.array(f0_contour)
    f0_plot[f0_plot == 0] = np.nan 
    
    plt.plot(times, f0_plot, label="F0 (Pitch)", color='b', linewidth=2)
    plt.title("Extracted Pitch Contour (Notice the Vibrato!)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True, alpha=0.5)
    plt.legend()

    # 2. RMS Energy 그래프
    plt.subplot(2, 1, 2)
    plt.plot(times, rms_contour, label="RMS Energy", color='r')
    plt.title("RMS Energy Contour")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy")
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # 1. 테스트용 오디오 준비 (직접 녹음한 wav 파일 경로를 넣으셔도 됩니다)
    test_sr = 24000
    test_audio_path = '/home/ysm/Vocal_Project/Vocal_Data/prompt.wav'
    if not os.path.exists(test_audio_path):
        generate_dummy_vocal(test_audio_path)

    # 2. Feature Extractor 초기화 및 실행
    print("========================================")
    print("보컬 Feature Extraction 시작...")
    extractor = VocalFeatureExtractor(test_audio_path, sr=test_sr)
    
    try:
        features = extractor.process_all()
        print("Extraction 완료!\n")
        
        # 3. 추출된 단일 통계값(Mean, Rate 등)만 터미널에 예쁘게 출력
        print("[추출된 보컬 요약 데이터]")
        # summary_data = {k: float(v) for k, v in features.items() if not isinstance(v, list)}
        summary_data = {}
        for k, v in features.items():
            if not isinstance(v, list):
                val = float(v)
                summary_data[k] = 0.0 if math.isnan(val) else val
        
        print(json.dumps(summary_data, indent=4, ensure_ascii=False))
        
        print("\n시계열 데이터(Contour) 길이 확인:")
        print(f" - F0 배열 길이: {len(features['f0_contour'])}")
        print(f" - F1,F2,F3 배열 길이: {len(features['f1_contour'])}")
        print(f" - RMS 배열 길이: {len(features['rms_contour'])}")
        print(f" - mfcc_mean 길이: {len(features['mfcc_mean'])}")
        
        print("\n다른 수치 데이터 확인:")
        print(f" - f0_mean: {features['f0_mean']}")
        print(f" - vibrato_rate_hz: {features['vibrato_rate_hz']}")
        print(f" - jitter_percent: {features['jitter_percent']}")
        print(f" - shimmer_percent: {features['shimmer_percent']}")
        print(f" - hnr_db: {features['hnr_db']}")
        print(f" - rms_mean: {features['rms_mean']}")
        print(f" - centroid_mean: {features['centroid_mean']}")
        
        # 4. 시각화 (그래프 그리기)
        plot_contour(features['f0_contour'], features['rms_contour'], sr=test_sr)
        
    except Exception as e:
        print(f"분석 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    main()