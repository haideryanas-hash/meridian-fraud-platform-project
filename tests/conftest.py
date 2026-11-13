import pytest

from fraudplat.config import load_settings
from fraudplat.data.simulator import simulate


@pytest.fixture(scope="session")
def small_events():
    s = load_settings()
    sim = dict(s["simulation"], n_customers=1500, n_days=40, n_rings=4, n_cnp_campaigns=8, n_ato_campaigns=6)
    lab = dict(s["labels"], chargeback_delay_days=[8, 15])
    return simulate(sim, lab)
