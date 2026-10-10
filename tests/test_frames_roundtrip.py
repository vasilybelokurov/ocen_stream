"""Round trip ICRS -> model frame -> ICRS for each solar frame (src/ocen_dm/streams/frames.py)."""
import numpy as np, pytest
from ocen_dm.streams.frames import FRAMES, OCEN_OBS, to_model, observables


@pytest.mark.parametrize("frame", list(FRAMES))
def test_roundtrip(frame):
    o = OCEN_OBS
    xv = to_model(o["ra"], o["dec"], 5.6, o["pmra"], o["pmdec"], o["vlos"], frame)
    b = observables(xv, frame)
    assert np.allclose([b["pmra"][0], b["pmdec"][0], b["vlos"][0], b["dist"][0]], [o["pmra"], o["pmdec"], o["vlos"], 5.6], atol=1e-8)
