import numpy as np
import py_sod_metrics

p_float = np.random.rand(256, 256).astype(np.float32)
t_binary = (np.random.rand(256, 256) > 0.5).astype(np.uint8)
t_255 = t_binary * 255

try:
    mae_metric = py_sod_metrics.MAE()
    fm_metric = py_sod_metrics.FmeasureV2()
    sm_metric = py_sod_metrics.Smeasure()
    em_metric = py_sod_metrics.Emeasure()
    
    mae_metric.step(p_float, t_255)
    fm_metric.step(p_float, t_255)
    sm_metric.step(p_float, t_255)
    em_metric.step(p_float, t_255)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
