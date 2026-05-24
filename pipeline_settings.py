import datetime
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Controls which pipeline stages are executed and optional I/O paths."""

    run_dns: bool = True
    run_solvers: bool = True
    run_projection: bool = True
    run_training_assembly: bool = True
    run_training: bool = True
    run_apriori: bool = True
    run_coupled: bool = True
    run_plotting: bool = True
    clip_pusuluri: bool = False
    clip_rajampeta: bool = False
    manual_path: str = ""

    def get_run_id(self, problem_name: str) -> str:
        """Return manual path if set, otherwise generate a timestamped run ID."""
        if self.manual_path:
            return self.manual_path
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        return f"run_{problem_name}_{timestamp}"

    @classmethod
    def all_stages(cls, manual_path: str = "") -> "PipelineConfig":
        """Run every stage (default full pipeline)."""
        return cls(manual_path=manual_path)

    @classmethod
    def coupled_only(cls, manual_path: str = "") -> "PipelineConfig":
        """Skip everything except coupled simulation and plotting."""
        return cls(
            run_solvers=False,
            run_projection=False,
            run_training_assembly=False,
            run_training=False,
            run_apriori=False,
            run_coupled=True,
            run_plotting=True,
            manual_path=manual_path,
        )

    @classmethod
    def coupled_only_clipped(cls, manual_path: str = "") -> "PipelineConfig":
        """Skip everything except coupled simulation and plotting."""
        return cls(
            run_solvers=False,
            run_projection=False,
            run_training_assembly=False,
            run_training=False,
            run_apriori=False,
            run_coupled=True,
            run_plotting=True,
            manual_path=manual_path,
            clip_pusuluri=True,
            clip_rajampeta=True,
        )

    @classmethod
    def all_stages_clipped(cls, manual_path: str = "") -> PipelineConfig:
        """Run all stages, the predictor model is clipped for stability."""
        return cls(manual_path=manual_path, clip_pusuluri=True, clip_rajampeta=True)

    @classmethod
    def all_but_dns_clipped(cls, manual_path: str = "") -> PipelineConfig:
        """Run all stages except DNS, the predictor model is clipped for stability."""
        return cls(
            manual_path=manual_path,
            run_dns=False,
            clip_pusuluri=True,
            clip_rajampeta=True,
        )

    @classmethod
    def only_plot(cls, manual_path: str) -> PipelineConfig:
        """Run only the plotting, requires manual path."""
        return cls(
            run_solvers=False,
            run_projection=False,
            run_training_assembly=False,
            run_training=False,
            run_apriori=False,
            run_coupled=False,
            run_plotting=True,
            manual_path=manual_path,
        )
