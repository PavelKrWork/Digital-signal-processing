import numpy as np
import librosa
import torch
import torchmetrics
# import pesq  # Дополнительная установка: pip install pesq

# SNR микшера
def mixer(original, noise, snr_db):
    """
    Смешивает оригинальный сигнал с шумом по заданному SNR (в дБ)
    """
    if len(noise) > len(original):
        noise = noise[:len(original)]
    elif len(noise) < len(original):
        raise ValueError("Шумовой сигнал должен быть не короче оригинального")
    
    original = original / np.max(np.abs(original))
    noise = noise / np.max(np.abs(noise))
    
    power_original = np.mean(original**2)
    power_noise = np.mean(noise**2)
    
    snr_linear = 10**(snr_db / 10)
    scale_factor = np.sqrt(power_original / (snr_linear * power_noise))
    
    noise_scaled = noise * scale_factor
    mixed = original + noise_scaled
    mixed = mixed / np.max(np.abs(mixed))
    
    return mixed

def evaluate_metrics(clean, mixed, sr=16000):
    """
    Вычисляет метрики качества для смешанного сигнала
    """
    clean_t = torch.from_numpy(clean).float()
    mixed_t = torch.from_numpy(mixed).float()
    
    # SDR
    sdr_metric = torchmetrics.audio.SignalDistortionRatio()
    sdr = sdr_metric(mixed_t, clean_t).item()
    
    # SI-SDR
    si_sdr_metric = torchmetrics.audio.ScaleInvariantSignalDistortionRatio()
    si_sdr = si_sdr_metric(mixed_t, clean_t).item()
    
    # # PESQ (используем библиотеку pesq)
    # try:
    #     pesq_score = pesq(sr, clean, mixed, 'wb')  # 'wb' для wide-band (16kHz)
    # except:
    #     # Если PESQ не работает, пробуем resample до 16kHz
    #     if sr != 16000:
    #         clean_16k = librosa.resample(clean, orig_sr=sr, target_sr=16000)
    #         mixed_16k = librosa.resample(mixed, orig_sr=sr, target_sr=16000)
    #         pesq_score = pesq(16000, clean_16k, mixed_16k, 'wb')
    #     else:
    #         pesq_score = float('nan')  # Если все равно ошибка
    
    return sdr, si_sdr

def main():
    # Загружаем файлы (пример)
    clean_audio, sr = librosa.load("gt.wav", sr=None)
    noise_audio, _ = librosa.load("noise.wav", sr=sr)
    
    snr_values = [-5, 0, 5, 10]
    results = []
    
    for snr in snr_values:
        mixed = mixer(clean_audio, noise_audio, snr)
        sdr, si_sdr = evaluate_metrics(clean_audio, mixed, sr)
        
        # Примерные MOS оценки
        mos_map = {-5: 1.5, 0: 2.5, 5: 3.5, 10: 4.2}
        
        results.append({
            "SNR": snr,
            "SDR": sdr,
            "SI-SDR": si_sdr,
            # "PESQ": pesq_score,
            "MOS": mos_map[snr]
        })
    
    # Вывод таблицы
    print("|    Файл     | SNR (dB) | SDR (dB) | SI-SDR (dB) | MOS |")
    print("|-------------|----------|----------|-------------|-----|")
    for i, res in enumerate(results):
        print(f"| mixed_{i+1}.wav |    {res['SNR']}    |    {res['SDR']:.2f}    |  {res['SI-SDR']:.2f}  | {res['MOS']:.1f} |")

if __name__ == "__main__":
    main()