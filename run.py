from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any, Dict


def _build_model_kwargs(args: argparse.Namespace, model_cls: type) -> Dict[str, Any]:
    """Build kwargs compatible with RobotMission signature."""
    candidate_kwargs = {
        "width": args.width,
        "height": args.height,
        "n_green_robots": args.n_green_robots,
        "n_yellow_robots": args.n_yellow_robots,
        "n_red_robots": args.n_red_robots,
        "n_green_wastes": args.n_green_wastes,
        "seed": args.seed,
        "verbose": args.verbose,
    }

    signature = inspect.signature(model_cls)
    accepted = set(signature.parameters.keys())
    return {k: v for k, v in candidate_kwargs.items() if k in accepted and v is not None}


def _count_wastes(model: Any) -> Dict[str, int]:
    """Count waste objects by waste_type in the grid."""
    counts = {"green": 0, "yellow": 0, "red": 0}
    grid = getattr(model, "grid", None)
    if grid is None or not hasattr(grid, "coord_iter"):
        return counts

    for _, x, y in grid.coord_iter():
        for obj in grid.get_cell_list_contents([(x, y)]):
            waste_type = getattr(obj, "waste_type", None)
            if waste_type in counts:
                counts[waste_type] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RobotMission simulation in batch mode.")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps.")
    parser.add_argument("--width", type=int, default=20, help="Grid width (if model supports it).")
    parser.add_argument("--height", type=int, default=10, help="Grid height (if model supports it).")
    parser.add_argument("--n-green-robots", type=int, dest="n_green_robots", help="Green robot count.")
    parser.add_argument("--n-yellow-robots", type=int, dest="n_yellow_robots", help="Yellow robot count.")
    parser.add_argument("--n-red-robots", type=int, dest="n_red_robots", help="Red robot count.")
    parser.add_argument("--n-green-wastes", type=int, dest="n_green_wastes", help="Initial green waste count.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Print waste counts at each step.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from model import RobotMission
    except Exception as exc:  # pragma: no cover
        print("Could not import RobotMission from model.py.")
        print(f"Import error: {exc}")
        return 1

    kwargs = _build_model_kwargs(args, RobotMission)
    model = RobotMission(**kwargs)

    if not hasattr(model, "step"):
        print("RobotMission does not define a step() method.")
        return 1

    for step in range(1, args.steps + 1):
        model.step()
        if args.verbose:
            counts = _count_wastes(model)
            print(
                f"step={step} green={counts['green']} "
                f"yellow={counts['yellow']} red={counts['red']}"
            )

    final_counts = _count_wastes(model)
    print("Simulation finished.")
    print(
        f"Final wastes: green={final_counts['green']} "
        f"yellow={final_counts['yellow']} red={final_counts['red']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

