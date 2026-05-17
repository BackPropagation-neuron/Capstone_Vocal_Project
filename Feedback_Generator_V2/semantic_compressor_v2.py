import numpy as np

class VocalSemanticCompressorV2:
    def __init__(self, sr=22050, hop_length=512):
        # vocal_extractor.py의 기본 설정(sr=22050, hop_length=512)과 완벽히 동기화
        self.sr = sr
        self.hop_length = hop_length
        self.fps = sr / hop_length
        
        # [가이드라인 업데이트] Extractor가 제공하는 모든 데이터를 해석할 수 있는 렌즈 제공
        self.reference_guide = {
            "HNR (Harmonics-to-Noise Ratio)": "15~20dB 이상이면 안정적인 성대 접촉, 10dB 미만이면 기식음(바람 새는 소리) 및 접촉 불량 의심.",
            "Jitter (주파수 변동률)": "1.0% 미만이 정상. 2.0% 이상일 경우 후두 긴장이나 성대 진동의 불규칙성(거친 소리) 의심.",
            "Shimmer (진폭 변동률)": "3.0% 미만이 정상. 5.0% 이상일 경우 호흡 압력의 불안정성 의심.",
            "Vibrato (비브라토)": "숙련된 가수의 자연스러운 비브라토는 보통 5.0Hz ~ 7.0Hz 사이에서 형성됨.",
            "Centroid (소리의 무게중심)": "수치가 높을수록 밝고 날카로운 음색, 낮을수록 어둡고 무거운 음색.",
            "Formants (F1/F2/F3)": "F1은 턱의 개폐(입 크기), F2는 혀의 전후 위치를 나타냄. F3는 소리의 명료도 및 뚫고 나오는 소리(Singer's Formant)와 연관됨."
        }

    def _format_global_stats(self, features):
        """전반적인 음향 통계를 포맷팅합니다."""
        lines = ["### 1. 전반적인 음향 통계 (Global Acoustic Statistics)"]
        lines.append("> [LLM을 위한 해석 가이드]")
        for key, guide in self.reference_guide.items():
            lines.append(f"> * {key}: {guide}")
        
        lines.append("\n[사용자 측정 수치]")
        # vocal_extractor.py 및 DB 모델의 컬럼명과 100% 일치
        f0_mean = features.get('f0_mean', 0.0)
        hnr = features.get('hnr_db', 0.0)
        jitter = features.get('jitter_percent', 0.0)
        shimmer = features.get('shimmer_percent', 0.0)
        vibrato = features.get('vibrato_rate_hz', 0.0)
        centroid_mean = features.get('centroid_mean', 0.0)
        
        lines.append(f"- 평균 음고(F0 Mean): {f0_mean:.1f} Hz")
        lines.append(f"- 조화음 대 잡음비(HNR): {hnr:.2f} dB")
        lines.append(f"- 주파수 변동률(Jitter): {jitter:.2f} %")
        lines.append(f"- 진폭 변동률(Shimmer): {shimmer:.2f} %")
        lines.append(f"- 비브라토 속도(Vibrato Rate): {vibrato:.1f} Hz")
        lines.append(f"- 소리 무게중심(Centroid): {centroid_mean:.1f} Hz")
        
        return "\n".join(lines)

    def _analyze_time_frames(self, features, chunk_sec=2.0):
        """시계열 데이터를 N초 단위 구간으로 나누어 평균과 변동성을 요약합니다."""
        lines = [f"\n### 2. 구간별 통계 분석 (Timeline Analysis - {chunk_sec}초 단위)"]
        
        # vocal_extractor.py의 extract_formants_lpc() 및 extract_spectral_features() 결과 가져오기
        f0_contour = features.get('f0_contour', [])
        f1_contour = features.get('f1_contour', [])
        f2_contour = features.get('f2_contour', [])
        f3_contour = features.get('f3_contour', []) # [추가됨] F3 데이터 활용
        rms_contour = features.get('rms_contour', [])
        
        if not f0_contour:
            return lines[0] + "\n- 유효한 시계열 데이터가 없습니다."

        frames_per_chunk = int(self.fps * chunk_sec)
        # 배열들의 길이가 미세하게 다를 수 있으므로 가장 짧은 길이를 기준으로 삼음 (안전장치)
        total_frames = min(len(f0_contour), len(f1_contour), len(rms_contour))
        
        for start_idx in range(0, total_frames, frames_per_chunk):
            end_idx = min(start_idx + frames_per_chunk, total_frames)
            
            # 발성이 있는(무음 0.0이 아닌) 유효 프레임만 필터링
            f0_chunk = [x for x in f0_contour[start_idx:end_idx] if x > 0]
            f1_chunk = [x for x in f1_contour[start_idx:end_idx] if x > 0]
            f2_chunk = [x for x in f2_contour[start_idx:end_idx] if x > 0]
            f3_chunk = [x for x in f3_contour[start_idx:end_idx] if x > 0]
            rms_chunk = [x for x in rms_contour[start_idx:end_idx] if x > 0]
            
            # 피치 데이터가 아예 없는 무음 구간은 스킵
            if not f0_chunk:
                continue 
                
            start_time = start_idx / self.fps
            end_time = end_idx / self.fps
            
            # 구간별 통계 산출 (빈 배열일 경우 0.0 반환)
            f0_mean = np.mean(f0_chunk)
            f0_std = np.std(f0_chunk)
            f1_mean = np.mean(f1_chunk) if f1_chunk else 0.0
            f2_mean = np.mean(f2_chunk) if f2_chunk else 0.0
            f3_mean = np.mean(f3_chunk) if f3_chunk else 0.0
            rms_mean = np.mean(rms_chunk) if rms_chunk else 0.0
            
            lines.append(f"[{start_time:.1f}초 ~ {end_time:.1f}초]")
            lines.append(f"  - F0 (Pitch): 평균 {f0_mean:.1f}Hz (변동폭: ±{f0_std:.1f}Hz)")
            lines.append(f"  - Formants: F1 {f1_mean:.0f}Hz / F2 {f2_mean:.0f}Hz / F3 {f3_mean:.0f}Hz")
            lines.append(f"  - RMS (Energy): 평균 {rms_mean:.4f}")
            
        return "\n".join(lines)

    def compress_to_semantic_text(self, features):
        """VocalExtractor의 출력 dict를 받아 최종 LLM 프롬프트용 텍스트를 반환합니다."""
        global_text = self._format_global_stats(features)
        timeline_text = self._analyze_time_frames(features, chunk_sec=2.0)
        return f"{global_text}\n{timeline_text}"

# ==========================================
# 실행 테스트 (Extractor와 연동 시뮬레이션)
# ==========================================
if __name__ == "__main__":
    compressor = VocalSemanticCompressorV2()
    
    # DB나 Extractor에서 막 뽑아낸 형태의 데이터
    mock_extractor_features = {
        'f0_mean': 440.5,
        'vibrato_rate_hz': 5.8,
        'jitter_percent': 1.5,
        'shimmer_percent': 3.2,
        'hnr_db': 18.5,
        'rms_mean': 0.12,
        'centroid_mean': 2200.0,
        
        # 약 4초 분량 (43fps * 4 = 172 프레임)의 시계열 배열
        'f0_contour': [440.0]*86 + [450.0]*86,
        'f1_contour': [500.0]*86 + [400.0]*86,
        'f2_contour': [1500.0]*86 + [1600.0]*86,
        'f3_contour': [2800.0]*86 + [2400.0]*86, # 후반부에 F3가 떨어져 소리가 먹먹해지는 상황
        'rms_contour': [0.15]*86 + [0.08]*86
    }
    
    # 변환 실행
    final_prompt_text = compressor.compress_to_semantic_text(mock_extractor_features)
    print(final_prompt_text)