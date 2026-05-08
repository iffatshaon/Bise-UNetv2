import numpy as np
import py_sod_metrics

p_float = np.random.rand(256, 256).astype(np.float32)
t_binary = (np.random.rand(256, 256) > 0.5).astype(np.uint8)

fm_metric = py_sod_metrics.Fmeasure()
fm_metric.step(p_float, t_binary)
print("Float Pred, Binary Target:", fm_metric.get_results()["fm"]["curve"].max())

t_255 = t_binary * 255
fm_metric2 = py_sod_metrics.Fmeasure()
fm_metric2.step(p_float, t_255)
print("Float Pred, 255 Target:", fm_metric2.get_results()["fm"]["curve"].max())

p_255 = (p_float * 255).astype(np.uint8)
fm_metric3 = py_sod_metrics.Fmeasure()
fm_metric3.step(p_255, t_255)
print("255 Pred, 255 Target:", fm_metric3.get_results()["fm"]["curve"].max())

