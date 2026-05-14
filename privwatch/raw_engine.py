import time
from privwatch.inference import run_inference


def raw_engine(cap):
    start = time.time()

    result = run_inference(cap)

    end = time.time()

    return result, round(end - start, 2), True  # stored = True
