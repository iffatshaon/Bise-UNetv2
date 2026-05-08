import numpy as np
import py_sod_metrics

p = np.random.rand(256, 256).astype(np.float32)
t = (np.random.rand(256, 256) > 0.5).astype(np.uint8)

fm_metric = py_sod_metrics.Fmeasure()
fm_metric.step(p, t)
res = fm_metric.get_results()
print(res)
