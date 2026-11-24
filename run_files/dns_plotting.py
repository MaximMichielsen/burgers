from pathlib import Path

from setup.constants import INTERVALS
from utils.profile_plotting import (
    plot_profile_comparisons_dns_filtered_vs_reference,
    plot_profile_comparisons_dns_filtered_vs_unfiltered,
)

root = Path(__file__).resolve().parent.parent

# filtered dns vs tcf reference
dns_references_dir = root / "dns_data" / "intervals_stat"
i = 0
for t_start, t_end in INTERVALS:
    interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
    if i == 0:
        dns_dir_unfiltered = (
            root / "runs" / "DNS" / "filter_comparison" / "unfiltered" / interval_dir
        )
        plot_profile_comparisons_dns_filtered_vs_unfiltered(
            intervals_dir=dns_references_dir,
            dns_dir=dns_dir_unfiltered,
            t_start=t_start,
        )
    i += 1

    dns_dir_filtered = root / "runs" / "DNS" / "filtered" / interval_dir

    plot_profile_comparisons_dns_filtered_vs_reference(
        intervals_dir=dns_references_dir, dns_dir=dns_dir_filtered, t_start=t_start
    )
