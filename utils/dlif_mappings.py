"""Shared DLIF / MLEF mapping utilities used by both classification and survival helpers."""


def outcome_to_dlif(outcome_str: str) -> str:
    """Convert MLEF outcome names to DLIF format."""
    mapping = {
        'OS_24': 'os_months_24',
        'OS_6': 'os_months_6',
        'DCR': 'DCR'
    }
    return mapping.get(outcome_str, outcome_str)


def map_dlif_to_mlef_modality(dlif_name: str) -> str:
    """
    Map DLIF modality names to MLEF-style names.
    DLIF: rwd, rwd_dp, rwd_radfm, rwd_radpy, rwd_radfm_dp, rwd_radpy_dp
    MLEF: RWD, RWD_DP, RWD_FMRAD, RWD_PYRAD, RWD_DP_FMRAD, RWD_DP_PYRAD
    """
    mapping = {
        'rwd': 'CB',
        'rwd_dp': 'CB_DP',
        'rwd_radfm': 'CB_FMRAD',
        'rwd_radpy': 'CB_PYRAD',
        'rwd_radfm_dp': 'CB_DP_FMRAD',
        'rwd_radpy_dp': 'CB_DP_PYRAD',
        'rwd_radfm_dp_genomics': 'CB_DP_FMRAD_GENOMICS',
        'rwd_radpy_dp_genomics': 'CB_DP_PYRAD_GENOMICS',
    }
    return mapping.get(dlif_name.lower(), dlif_name.upper())


def map_mlef_to_dlif_modality(mlef_name: str) -> str:
    """
    Map MLEF modality names to DLIF-style names.
    """
    mapping = {
        'CB': 'rwd',
        'CB_DP': 'rwd_dp',
        'CB_FMRAD': 'rwd_radfm',
        'CB_PYRAD': 'rwd_radpy',
        'CB_DP_FMRAD': 'rwd_radfm_dp',
        'CB_DP_PYRAD': 'rwd_radpy_dp',
        'CB_DP_FMRAD_GENOMICS': 'rwd_radfm_dp_genomics',
        'CB_DP_PYRAD_GENOMICS': 'rwd_radpy_dp_genomics',
    }
    return mapping.get(mlef_name.upper(), mlef_name.lower())
