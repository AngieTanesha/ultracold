import numpy as np

# Each value of r0, used with the corresponding value of V0. will give a
# scattering length of around 1E8 i.e. approximately unitarity
RV = np.array([
    [0.05 / np.sqrt(2), -536.8013763987018],
    [0.04 / np.sqrt(2), -838.7522587399022],
    [0.03 / np.sqrt(2), -1491.115421959569],
    [0.02 / np.sqrt(2), -3355.009213091155],
    [0.01 / np.sqrt(2), -13420.03441529311],
])

R0 = RV[:, 0]
V0 = RV[:, 1]
