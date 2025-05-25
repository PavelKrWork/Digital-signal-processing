import numpy as np

def align_signals(ref, target):
    # Нормализация (по максимуму, чтобы сравнение было корректным)
    ref = ref / np.max(np.abs(ref))
    target = target / np.max(np.abs(target))

    # Кросс-корреляция (mode='full' ищет все возможные сдвиги)
    corr = np.correlate(target, ref, mode='full')
    delay = np.argmax(corr) - len(ref) + 1

    print(f"Сдвиг: {delay} отсчетов")

    # Сдвигаем сигнал target относительно ref
    if delay > 0:
        aligned = np.pad(target, (delay, 0), mode='constant')
    else:
        aligned = target[-delay:]

    # Усечение или дополнение: выравниваем длины (если нужно)
    min_len = min(len(ref), len(aligned))
    ref_aligned = ref[:min_len]
    target_aligned = aligned[:min_len]

    return ref_aligned, target_aligned, delay