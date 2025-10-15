from pathlib import Path
from typing import Optional


def describe_meteostat() -> str:
    return (
        "meteostat: météo historique (stations Paris).\n"
        "Use for training-time features; at inference, rely on climatologies only."
    )


def describe_era5() -> str:
    return (
        "ERA5 reanalysis: atmospheric variables (e.g., wind, pressure, PBLH).\n"
        "Use historical reanalysis to engineer features; no test-period data."
    )


def describe_workalendar() -> str:
    return (
        "workalendar (France): official/public holidays for France.\n"
        "Use to flag holidays/bridges; derive features for calendar effects."
    )


def describe_school_zone_c() -> str:
    return (
        "French school holidays (Zone C - Paris) via MENJ calendars (public).\n"
        "Use to mark school breaks; potential impact on traffic/emissions."
    )


def describe_traffic() -> str:
    return (
        "Traffic archives: public city reports or datasets if available.\n"
        "Engineer congestion proxies; ensure historical-only usage."
    )


def load_stub(*_, **__):
    raise NotImplementedError(
        "Implement specific loaders and mapping to hourly Paris timestamps as needed."
    )


