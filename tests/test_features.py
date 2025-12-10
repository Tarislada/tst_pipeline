from tst.features import extract_hard_features, apply_rule_and_uncertainty
import numpy as np

def test_extract_shapes():
    T=300; fps=30
    area = np.sin(np.linspace(0, 10, T)).astype('float32')*10+100
    sim = np.clip(np.random.rand(T).astype('float32'),0,1)
    df = extract_hard_features(area, sim, fps=fps, win_s=1.0)
    assert len(df) == T//fps
    df2 = apply_rule_and_uncertainty(df)
    assert 'state_rule' in df2.columns
