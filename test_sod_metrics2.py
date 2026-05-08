import numpy as np
import py_sod_metrics

p_float = np.random.rand(256, 256).astype(np.float32)
t_binary = (np.random.rand(256, 256) > 0.5).astype(np.uint8)
t_255 = t_binary * 255

fm_metric = py_sod_metrics.FmeasureV2()
fm_metric.add_handler("fm", py_sod_metrics.FmeasureHandler(with_dynamic=True, with_adaptive=False, with_binary=False))

fm_metric.step(p_float, t_255)
res = fm_metric.get_results()

print("Fmeasure Curve Max:", res["fm"]["dynamic"].max())
