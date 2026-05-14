import time
from privwatch.privacy_inference import run_privacy_inference


def privacy_engine(cap):
    start = time.time()

    result = run_privacy_inference(cap)

    end = time.time()

    return result, round(end - start, 2), False  # stored = False
