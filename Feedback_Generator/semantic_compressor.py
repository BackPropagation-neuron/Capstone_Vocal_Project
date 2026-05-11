import numpy as np

class VocalSemanticCompressor:
    def __init__(self, sr=24000, hop_length=512):
        self.fps = sr / hop_length

    def _analyze_global_features(self, features):
        insights = ["### 1. 전반적인 발성 상태 (Global Analysis)"]
        
        hnr = features.get('hnr_db', 0)
        if hnr < 10:
            insights.append("- [음색] 성대 접촉이 상대적으로 약해 호흡이 섞인 톤(Breathy)입니다.")
        elif hnr > 20:
            insights.append("- [음색] 성대 접촉이 강해 밀도 있고 단단한 톤입니다.")

        centroid = features.get('centroid_mean', 0)
        if centroid > 3000:
            insights.append("- [음색] 소리의 무게중심(Centroid)이 높아 밝고 고음역이 강조된 음색입니다.")
        elif 0 < centroid < 1200:
            insights.append("- [음색] 소리의 무게중심이 낮아 비교적 무겁고 어두운 음색입니다.")

        jitter = features.get('jitter_percent', 0)
        if jitter > 2.0:
            insights.append("- [긴장도] 성대 진동(Jitter) 불규칙성이 감지됩니다. 발성 시 후두 긴장도가 높을 수 있습니다.")

        shimmer = features.get('shimmer_percent', 0)
        if shimmer > 5.0:
            insights.append("- [호흡] 진폭(Shimmer) 불규칙성이 감지됩니다. 호흡 압력이 다소 불안정합니다.")

        vibrato = features.get('vibrato_rate_hz', 0)
        if vibrato == 0:
            insights.append("- [기교] 바이브레이션 없이 직선적인 톤(Straight tone)을 유지합니다.")
        elif 4.5 <= vibrato <= 6.5:
            insights.append("- [기교] 초당 약 5~6회 수준의 규칙적인 바이브레이션이 존재합니다.")
        elif vibrato > 7.0:
            insights.append("- [기교] 7Hz 이상의 빠른 바이브레이션 패턴이 나타납니다.")

        mfcc = features.get('mfcc_mean', [])
        if len(mfcc) > 0:
            insights.append("- [음색 지문] 고유 보컬 톤(MFCC) 데이터가 정상적으로 매핑되었습니다.")
        
        return "\n".join(insights)

    def _analyze_time_frames(self, features, chunk_sec=5.0):
        insights = ["\n### 2. 시간대별 상세 진단 (Timeline Analysis)"]
        
        f0_contour = np.array(features.get('f0_contour', []))
        f1_contour = np.array(features.get('f1_contour', []))
        rms_contour = np.array(features.get('rms_contour', []))
        f2_contour = np.array(features.get('f2_contour', []))
        f3_contour = np.array(features.get('f3_contour', []))
        
        total_frames = len(f0_contour)
        frames_per_chunk = int(self.fps * chunk_sec)
        
        for start_idx in range(0, total_frames, frames_per_chunk):
            end_idx = min(start_idx + frames_per_chunk, total_frames)
            start_time = start_idx / self.fps
            end_time = end_idx / self.fps
            
            # --- 유효값 필터링 및 마스크 생성 (버그 수정 부분) ---
            f0_chunk = f0_contour[start_idx:end_idx]
            valid_mask = f0_chunk > 0 # 피치가 0보다 큰(유의미한 소리가 나는) 구간 마스킹
            
            f0_valid = f0_chunk[valid_mask]
            
            # 유효한 소리가 0.5초 미만인 구간은 생략
            if len(f0_valid) < (self.fps * 0.5):
                continue
                
            rms_chunk = rms_contour[start_idx:end_idx]
            rms_valid = rms_chunk[rms_chunk > 0.001]
            
            status_text = []
            
            # 1. 음정 (F0)
            chunk_f0_mean = np.mean(f0_valid)
            if np.std(f0_valid) > 25.0:
                status_text.append("음정 흔들림(Pitch Instability)")
            else:
                status_text.append("음정 안정적")
                
            # 2. 강약 조절 (RMS)
            if len(rms_valid) > 0:
                if (np.max(rms_valid) - np.min(rms_valid)) < 0.05:
                    status_text.append("다이내믹스 변화 적음")
                else:
                    status_text.append("다이내믹스 폭 넓음")
                    
            # 3. 객관적 조음 공간 (F1: 턱의 벌어짐 / 모음의 높낮이)
            f1_chunk = f1_contour[start_idx:end_idx]
            # 배열 길이가 다를 수 있는 예외를 방지하기 위해 마스크 길이를 맞춤
            safe_mask = valid_mask[:len(f1_chunk)] 
            f1_valid = f1_chunk[safe_mask]
            
            if len(f1_valid) > 0:
                f1_mean = np.mean(f1_valid)
                if f1_mean < 350:
                    status_text.append("구강 공간 좁음(고모음 성향)")
                elif f1_mean > 700:
                    status_text.append("구강 공간 넓음(저모음 성향)")
                    
            # 4. 객관적 혀의 위치 (F2: 전설/후설 모음)
            f2_chunk = f2_contour[start_idx:end_idx]
            safe_mask_f2 = valid_mask[:len(f2_chunk)]
            f2_valid = f2_chunk[safe_mask_f2]
            
            if len(f2_valid) > 0:
                f2_mean = np.mean(f2_valid)
                if f2_mean > 1800:
                    status_text.append("혀 위치 전진(Front Vowel 성향)")
                elif f2_mean < 1000:
                    status_text.append("혀 위치 후퇴(Back Vowel 성향)")

            # 5. 객관적 공명 특성 (F3: 가수 포먼트 대역)
            f3_chunk = f3_contour[start_idx:end_idx]
            safe_mask_f3 = valid_mask[:len(f3_chunk)]
            f3_valid = f3_chunk[safe_mask_f3]
            
            if len(f3_valid) > 0:
                f3_mean = np.mean(f3_valid)
                if f3_mean > 2500:
                    status_text.append("2.5kHz 이상 고주파 대역 에너지 높음(트웽 특성)")
                else:
                    status_text.append("고주파 대역 에너지 낮음")

            time_label = f"[{start_time:02.0f}초~{end_time:02.0f}초]"
            insight = f"{time_label} 평균 {chunk_f0_mean:.0f}Hz 대역 | {', '.join(status_text)}"
            insights.append(insight)
            
        if len(insights) == 1:
            insights.append("- 유효한 보컬 구간이 감지되지 않았습니다.")
            
        return "\n".join(insights)

    def compress_to_semantic_text(self, features):
        global_text = self._analyze_global_features(features)
        timeline_text = self._analyze_time_frames(features, chunk_sec=5.0)
        
        return f"{global_text}\n{timeline_text}"

# ==========================================
# 실행 예시 (테스트 블록)
# ==========================================
if __name__ == "__main__":
    # 총 500프레임 (약 10.6초 분량, 47fps 기준)의 가상 배열 생성
    # 0~150: 안정적 구간 / 150~400: 흔들리고 좁아지는 구간 / 400~500: 무음
    f0 = [440]*150 + [445, 430, 480, 410, 450]*50 + [0]*100
    rms = [0.1]*150 + [0.1, 0.11, 0.1, 0.1]*62 + [0.1, 0.1] + [0]*100
    f1 = [600]*150 + [300, 310, 305]*83 + [300] + [0]*100
    f2 = [1500]*150 + [900, 950, 920]*83 + [900] + [0]*100
    f3 = [2800]*150 + [2200, 2100, 2150]*83 + [2200] + [0]*100

    dummy_features = {
        'hnr_db': 8.5,
        'jitter_percent': 2.5,
        'shimmer_percent': 6.0,
        'vibrato_rate_hz': 0.0,
        'centroid_mean': 2500,
        'mfcc_mean': [-200, 80, -10, 20],
        'f0_contour': f0,
        'rms_contour': rms,
        'f1_contour': f1,
        'f2_contour': f2,
        'f3_contour': f3
    }
    
    compressor = VocalSemanticCompressor()
    semantic_result = compressor.compress_to_semantic_text(dummy_features)
    
    print("[LLM 진단용 최종 텍스트 추출 완료]\n")
    print(semantic_result)