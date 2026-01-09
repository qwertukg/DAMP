from __future__ import annotations

from typing import TYPE_CHECKING

import rerun as rr

if TYPE_CHECKING:
    from .damp_layout import Layout


def log_layout(layout: "Layout", *, path: str = "layout", step: int | None = None) -> None:
    if step is not None:
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("step", step)
    rr.log(f"{path}/image", rr.Image(layout.render_image()))

    positions = [(x, y) for y, x in layout.positions()]
    rr.log(
        f"{path}/points",
        rr.Points2D(
            positions,
            colors=layout.colors_rgb(),
            radii=0.45,
            labels=layout.labels(),
            show_labels=False,
        ),
        rr.AnyValues(values=layout.values()),
    )


def visualize_layout(
    layout: "Layout", *, app_id: str = "damp-layout", path: str = "layout"
) -> None:
    rr.init(app_id)
    rr.spawn()
    log_layout(layout, path=path)
