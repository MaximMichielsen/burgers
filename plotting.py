from functions import SolutionConfig

dns_settings = SolutionConfig(
    data_path=solver_data_path_dns,
    label="DNS",
    color="gray",
    linestyle="-",
    marker="",  # no marker for the reference curve
    alpha=0.7,
    mesh=mesh_dns,
    solution=dns_solution,
)
