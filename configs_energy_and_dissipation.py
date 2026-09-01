from pathlib import Path


def plotting_configs(paths) -> list[tuple[str, Path | None, str, str, float]]:
    """Standard plotting configurations for energy and dissipation plots."""
    _all_configs: list[tuple[str, Path | None, str, str, float]] = [
        ("DNS", paths.dns_data, "gray", "-", 1.8),
        ("Projection", paths.projection, "lightgreen", "-", 1.2),
        ("LES - A", paths.les_a_data, "tab:orange", "--", 1.4),
        ("LES - Shakib 1", paths.les_shakib_one_data, "royalblue", "--", 1.4),
        ("LES - NM", paths.les_nm_data, "gold", "-.", 1.4),
        ("LES - AVCG", paths.avc_data / "global", "royalblue", "-", 1.8),
        ("LES - AVCL", paths.avc_data / "gl_hybrid", "blueviolet", "--", 1.8),
    ]
    return _all_configs
