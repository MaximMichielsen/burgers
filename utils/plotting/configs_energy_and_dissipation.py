from pathlib import Path

from utils.pipeline_utils import RunPaths


#TODO: adjust configs

def plotting_configs(paths: RunPaths) -> list[tuple[str, Path | None, str, str, float]]:
    """Standard plotting configurations for energy and dissipation plots."""
    _all_configs: list[tuple[str, Path | None, str, str, float]] = [
        ("DNS", paths.dns_data, "gray", "-", 1.8),
        ("Projection", paths.projection, "lightgreen", "-", 1.2),
        ("LES - A", paths.les_a_data, "tab:orange", "--", 1.4),
        ("LES - Shakib 1", paths.les_tau_two_params_data, "royalblue", "--", 1.4),
        ("LES - Shakib 1", paths.les_tau_four_params_data, "royalblue", "--", 1.4),
        ("LES - NM", paths.les_nm_data, "gold", "-.", 1.4),
        ("LES - AVCG", paths.avc_data / "global", "royalblue", "-", 1.8),
        ("LES - AVCL", paths.avc_data / "gl_hybrid", "blueviolet", "--", 1.8),
        ("LES - ANN", paths.data_ann_path, "purple", "-", 1.8),
    ]
    return _all_configs
