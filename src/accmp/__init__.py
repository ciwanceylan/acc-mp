import os

if "ACCMP_USE_NUMBA_CACHE" not in os.environ or len(os.environ["ACCMP_USE_NUMBA_CACHE"]) != 1:
    os.environ["ACCMP_USE_NUMBA_CACHE"] = "1"
