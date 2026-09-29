from numba import hip

device = hip.detect()
print(device)
