import numpy as np
def get_coeffs(ampl_freq_resp, number_of_lanes = 32, flag = 1) -> np.array:
    # flag: 0 - mean, 1 - median
    res = np.zeros(number_of_lanes)
    elem_in_lane = len(ampl_freq_resp) // number_of_lanes
    for k in range(1, number_of_lanes):
        if flag == 0:
            res[k - 1] = np.median(ampl_freq_resp[(k - 1) * elem_in_lane : k * elem_in_lane])
        elif flag == 1:
            res[k - 1] = np.mean(ampl_freq_resp[(k - 1) * elem_in_lane : k * elem_in_lane])
        else:
            raise ValueError("Некорректное значение флага!")
    if flag == 0 : res[number_of_lanes - 1] = np.median(ampl_freq_resp[k * elem_in_lane : len(ampl_freq_resp)])
    elif flag == 1: res[number_of_lanes - 1] = np.mean(ampl_freq_resp[k * elem_in_lane : len(ampl_freq_resp)])
    else : raise ValueError("Некорректное значение флага!")

    return res