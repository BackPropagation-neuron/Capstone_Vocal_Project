import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import savgol_filter, find_peaks

class VocalFeatureExtractor:
    def __init__(self, audio_path, sr=22050):
        self.audio_path = audio_path
        self.sr = sr
        
        # 1. Librosa를 통한 오디오 로드 및 리샘플링
        self.y, self.sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # 파일 경로가 아닌, librosa가 로드한 numpy 배열(self.y)로 Praat 객체 생성
        self.sound = parselmouth.Sound(self.y, self.sr)

    def extract_pitch_and_vibrato(self):
            f0, voiced_flag, voiced_probs = librosa.pyin(self.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=self.sr)
            valid_f0 = f0[voiced_flag]
            vibrato_rate = 0.0

            if len(valid_f0) > 10:
                trend = savgol_filter(valid_f0, window_length=11, polyorder=2)
                detrended = valid_f0 - trend
                peaks, _ = find_peaks(detrended)
                if len(peaks) > 1:
                    hop_length = 512 / self.sr
                    avg_peak_interval = np.mean(np.diff(peaks)) * hop_length
                    vibrato_rate = 1.0 / avg_peak_interval if avg_peak_interval > 0 else 0

            # [추가된 부분] voiced_probs 배열의 평균값을 구합니다. 
            # NaN(계산 불가) 값이 있을 수 있으므로 np.nanmean을 사용하여 안전하게 평균을 냅니다.
            voiced_probs_mean = float(np.nanmean(voiced_probs)) if voiced_probs is not None and len(voiced_probs) > 0 else 0.0

            return {
                "f0_mean": float(np.nanmean(f0)) if len(valid_f0) > 0 else 0.0, 
                "f0_contour": np.nan_to_num(f0).tolist(), 
                "vibrato_rate_hz": vibrato_rate,
                "voiced_probs_mean": voiced_probs_mean  # [추가된 부분] 결과를 딕셔너리에 포함시킵니다!
            }

    def extract_formants_lpc(self):
        """전체 오디오 구간에 대해 프레임 단위로 Sliding Window LPC 분석 (Formant Contours)"""
        # 보컬 대역폭에 맞게 Pre-emphasis 적용
        y_preemp = librosa.effects.preemphasis(self.y)
        
        frame_length = int(self.sr * 0.05) # 50ms 프레임
        hop_length = 512 # 약 23ms 간격으로 이동 (F0 및 다른 특성과 싱크를 맞춤)
        
        # [배열 길이 동기화를 위한 Center Padding 추가]
        # F0, RMS와 길이가 완벽히 똑같이 나오도록 양끝에 빈 프레임을 덧붙여줍니다.
        y_padded = np.pad(y_preemp, (frame_length // 2, frame_length // 2), mode='reflect')
        
        # 오디오를 frame_length 크기로 쪼개서 2차원 배열로 만듦
        frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
        
        f1_contour, f2_contour, f3_contour = [], [], []
        n_poles = int(self.sr / 1000) + 2
        
        # 각 프레임마다 반복하며 Formant 추출
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # 무음이나 너무 작은 소리 프레임은 분석 생략 (연산 최적화 및 오류 방지)
            if np.max(np.abs(frame)) < 0.01:
                f1_contour.append(0.0)
                f2_contour.append(0.0)
                f3_contour.append(0.0)
                continue
                
            try:
                a = librosa.lpc(frame, order=n_poles)
                roots = np.roots(a)
                roots = roots[np.imag(roots) > 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formants = sorted(angles * (self.sr / (2 * np.pi)))
                
                f1_contour.append(formants[0] if len(formants) > 0 else 0.0)
                f2_contour.append(formants[1] if len(formants) > 1 else 0.0)
                f3_contour.append(formants[2] if len(formants) > 2 else 0.0)
            except:
                # LPC 연산 실패 시 0 처리
                f1_contour.append(0.0)
                f2_contour.append(0.0)
                f3_contour.append(0.0)
        
        return {
            "f1_contour": f1_contour,
            "f2_contour": f2_contour,
            "f3_contour": f3_contour
        }

    def extract_voice_quality(self):
        """Praat 엔진을 활용한 정밀 Jitter, Shimmer, HNR 추출"""
        # 서버 크래시 방지를 위한 try-except 안전망 추가
        try:
            pitch = call(self.sound, "To Pitch", 0.0, 75, 600)
            point_process = call([self.sound, pitch], "To PointProcess (cc)")
            
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([self.sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            harmonicity = call(self.sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            return {
                "jitter_percent": jitter * 100 if not np.isnan(jitter) else 0.0,
                "shimmer_percent": shimmer * 100 if not np.isnan(shimmer) else 0.0,
                "hnr_db": hnr if not np.isnan(hnr) else 0.0
            }
        except parselmouth.PraatError as e:
            # 유효한 피치를 찾지 못하는 등의 이유로 에러 발생 시 0.0 반환
            print(f"Praat Analysis Warning: {e}")
            return {"jitter_percent": 0.0, "shimmer_percent": 0.0, "hnr_db": 0.0}

    def extract_spectral_features(self):
        rms = librosa.feature.rms(y=self.y)[0]
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        return {"rms_mean": np.mean(rms), "rms_contour": rms.tolist(), "centroid_mean": np.mean(centroid), "mfcc_mean": np.mean(mfcc, axis=1).tolist()}

    def process_all(self):
        result = {}
        result.update(self.extract_pitch_and_vibrato())
        result.update(self.extract_formants_lpc())
        result.update(self.extract_voice_quality())
        result.update(self.extract_spectral_features())
        return result