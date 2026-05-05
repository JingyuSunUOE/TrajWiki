"""Hyper-parameter sweep support for retrieval and trajectory grid searches."""

from __future__ import annotations

from itertools import product

from trajpatch.config import RunConfig

from .runner import PipelineRunner


def run_grid(
    base_config: RunConfig,
    m_values: list[int],
    k_values: list[int],
    t_page_values: list[int] | None = None,
    neighbor_radius_values: list[int] | None = None,
    retrieval_expansion_mode_values: list[str] | None = None,
    r_values: list[int] | None = None,
) -> list[dict]:
    reports: list[dict] = []
    if t_page_values is None:
        t_page_values = [int(base_config.t_pages)]
    if neighbor_radius_values is None:
        neighbor_radius_values = [int(base_config.neighbor_radius)]
    if retrieval_expansion_mode_values is None:
        retrieval_expansion_mode_values = [str(base_config.retrieval_expansion_mode)]
    base_payload = (
        base_config.model_dump()
        if hasattr(base_config, "model_dump")
        else base_config.dict()  # type: ignore[attr-defined]
    )
    for m_value, t_page_value, k_value, neighbor_radius, retrieval_expansion_mode in product(
        m_values, t_page_values, k_values, neighbor_radius_values, retrieval_expansion_mode_values
    ):
        config = RunConfig(
            **{
                **base_payload,
                "m": m_value,
                "t_pages": t_page_value,
                "k": k_value,
                "neighbor_radius": neighbor_radius,
                "retrieval_expansion_mode": retrieval_expansion_mode,
                "database_path": None,
                "index_database_path": base_config.index_database_path,
                "output_dir": base_config.output_dir
                / (
                    f"sweep_m{m_value}_tp{t_page_value}_k{k_value}_nr{neighbor_radius}"
                    f"_mode_{retrieval_expansion_mode}"
                ),
            }
        )
        report = PipelineRunner(config).run()
        reports.append(
            {
                "m": m_value,
                "t_pages": t_page_value,
                "k": k_value,
                "neighbor_radius": neighbor_radius,
                "retrieval_expansion_mode": retrieval_expansion_mode,
                "metrics": report.metrics,
                "details": report.details,
            }
        )
    return reports
