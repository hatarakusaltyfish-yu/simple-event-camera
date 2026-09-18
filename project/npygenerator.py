import numpy as np

events = np.loadtxt(
    "result/decoded.txt",
    dtype=[
        ("x", np.int32),
        ("y", np.int32),
        ("p", np.uint8),
        ("t", np.float64)
    ]
)

np.save("result/events.npy", events)

print("\n" + "=" * 60)
print("events.npy saved!")
print("=" * 60)