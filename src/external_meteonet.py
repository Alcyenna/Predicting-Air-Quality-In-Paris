from pathlib import Path
from typing import Optional


def describe_meteonet_availability(root_dir: Optional[Path] = None) -> str:
    """
    Return a short description and pointers for integrating MeteoNet data.

    Note: MeteoNet covers 2016–2018; for this challenge, prefer deriving
    climatological hourly patterns rather than using any future-period data.
    """
    return (
        "MeteoNet (2016–2018) — Etalab Open Licence 2.0.\n"
        "Repo: https://github.com/meteofrance/meteonet\n"
        "Kaggle: https://www.kaggle.com/katerpillar/meteonet\n"
        "Suggested use: compute hourly/day-of-week climatologies for weather proxies."
    )


def load_meteonet_stub(*_, **__):
    """
    Placeholder loader. Implement if you decide to download MeteoNet locally.
    Should return a dataframe indexed by timestamp with weather features.
    """
    raise NotImplementedError(
        "Integrate MeteoNet by downloading data and mapping to Paris hourly timestamps."
    )


