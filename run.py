"""
Group: 14
Date: 16 March 2026
Members: Deodato V. Bastos Neto, Karina Musina
"""

from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from typing import Any, Dict, Iterable, List, Tuple


def _build_model_kwargs(args: argparse.Namespace, model_cls: type) -> Dict[str, Any]:
    """Build kwargs compatible with RobotMission signature."""
    requested_waste = args.n_waste
    if requested_waste is None:
        requested_waste = args.n_green_wastes
    if requested_waste is None:
        requested_waste = 30

    candidate_kwargs = {
        "width": args.width,
        "height": args.height,
        "n_robots": args.n_robots,
        "n_green_robots": args.n_green_robots,
        "n_yellow_robots": args.n_yellow_robots,
        "n_red_robots": args.n_red_robots,
        "n_green_wastes": requested_waste,
        "n_waste": requested_waste,
        "seed": args.seed,
        "verbose": args.verbose,
    }

    signature = inspect.signature(model_cls)
    accepted = set(signature.parameters.keys())
    return {k: v for k, v in candidate_kwargs.items() if k in accepted and v is not None}


def _iter_grid_objects(model: Any) -> Iterable[Any]:
    """Yield each object placed on the grid once."""
    grid = getattr(model, "grid", None)
    if grid is None or not hasattr(grid, "coord_iter"):
        return

    seen_ids = set()
    for item in grid.coord_iter():
        if not isinstance(item, tuple):
            continue

        if len(item) == 3:
            cell_content, x, y = item
        elif len(item) == 2 and isinstance(item[1], tuple) and len(item[1]) == 2:
            cell_content, (x, y) = item
        else:
            continue

        objects: List[Any]
        if isinstance(cell_content, list):
            objects = cell_content
        else:
            objects = grid.get_cell_list_contents([(x, y)])

        for obj in objects:
            obj_id = id(obj)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            yield obj


def _count_wastes(model: Any) -> Dict[str, int]:
    """Count waste objects by waste_type in the grid."""
    counts = {"green": 0, "yellow": 0, "red": 0}

    for obj in _iter_grid_objects(model) or []:
        waste_type = getattr(obj, "waste_type", None)
        if waste_type in counts:
            counts[waste_type] += 1

    return counts


def _count_robots(model: Any) -> Dict[str, int]:
    """Count robot objects by class name."""
    counts = {
        "GreenAgent": 0,
        "YellowAgent": 0,
        "RedAgent": 0,
        "greenAgent": 0,
        "yellowAgent": 0,
        "redAgent": 0,
    }
    for obj in _iter_grid_objects(model) or []:
        name = obj.__class__.__name__
        if name in counts:
            counts[name] += 1
    g = counts["GreenAgent"] + counts["greenAgent"]
    y = counts["YellowAgent"] + counts["yellowAgent"]
    r = counts["RedAgent"] + counts["redAgent"]
    return {"green": g, "yellow": y, "red": r}


def _preflight_checks(model: Any) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) after checking key model capabilities."""
    errors: List[str] = []
    warnings: List[str] = []

    if not callable(getattr(model, "step", None)):
        errors.append("RobotMission has no callable step() method.")

    if not callable(getattr(model, "do", None)):
        warnings.append("RobotMission has no callable do(agent, action) method.")

    grid = getattr(model, "grid", None)
    if grid is None:
        errors.append("RobotMission has no grid attribute.")
    else:
        if not callable(getattr(grid, "get_neighborhood", None)):
            errors.append("model.grid has no callable get_neighborhood(...).")
        if not callable(getattr(grid, "get_cell_list_contents", None)):
            errors.append("model.grid has no callable get_cell_list_contents(...).")
        if not callable(getattr(grid, "move_agent", None)):
            warnings.append("model.grid has no callable move_agent(...).")

    return errors, warnings


def _print_counts(step: int, model: Any) -> None:
    wastes = _count_wastes(model)
    robots = _count_robots(model)
    print(
        f"step={step} "
        f"robots(g={robots['green']},y={robots['yellow']},r={robots['red']}) "
        f"wastes(g={wastes['green']},y={wastes['yellow']},r={wastes['red']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RobotMission simulation in batch mode.")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps.")
    parser.add_argument("--width", type=int, default=20, help="Grid width (if model supports it).")
    parser.add_argument("--height", type=int, default=10, help="Grid height (if model supports it).")
    parser.add_argument("--n-robots", type=int, dest="n_robots", help="Total robot count (optional).")
    parser.add_argument("--n-green-robots", type=int, dest="n_green_robots", help="Green robot count.")
    parser.add_argument("--n-yellow-robots", type=int, dest="n_yellow_robots", help="Yellow robot count.")
    parser.add_argument("--n-red-robots", type=int, dest="n_red_robots", help="Red robot count.")
    parser.add_argument("--n-green-wastes", type=int, dest="n_green_wastes", help="Initial green waste count.")
    parser.add_argument(
        "--n-waste",
        type=int,
        dest="n_waste",
        default=30,
        help="Initial waste count (generic, default: 30).",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Print counts during simulation.")
    parser.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="When verbose, print a report every N steps.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run model initialization and checks only, without stepping.",
    )
    parser.add_argument(
        "--debug-traceback",
        action="store_true",
        help="Print full traceback on errors.",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help="Model class name in model.py (default: auto-detect).",
    )
    return parser.parse_args()


def _select_model_class(module: Any, class_name: str | None) -> type | None:
    if class_name:
        return getattr(module, class_name, None)

    for name in ("RobotMission", "RobotMissionModel"):
        cls = getattr(module, name, None)
        if cls is not None and inspect.isclass(cls):
            return cls
    return None


def main() -> int:
    args = parse_args()

    try:
        import model as model_module
    except Exception as exc:  # pragma: no cover
        print(f"Import error: {exc}")
        if args.debug_traceback:
            traceback.print_exc()
        return 1

    model_cls = _select_model_class(model_module, args.model_class)
    if model_cls is None:
        class_names = [
            name
            for name, obj in inspect.getmembers(model_module, inspect.isclass)
            if obj.__module__ == model_module.__name__
        ]
        print("Could not find model class in model.py.")
        print("Expected class name: RobotMission or RobotMissionModel.")
        print(f"Available classes: {class_names}")
        return 1

    kwargs = _build_model_kwargs(args, model_cls)
    print(f"Initializing {model_cls.__name__} with kwargs: {kwargs}")
    try:
        model = model_cls(**kwargs)
    except Exception as exc:
        print(f"Failed to initialize {model_cls.__name__}.")
        print(f"Initialization error: {exc}")
        if args.debug_traceback:
            traceback.print_exc()
        return 1

    errors, warnings = _preflight_checks(model)
    for warning in warnings:
        print(f"Warning: {warning}")
    for error in errors:
        print(f"Error: {error}")
    if errors:
        return 1

    print("Preflight checks passed.")
    _print_counts(step=0, model=model)

    if args.check_only:
        print("Check-only mode: simulation not executed.")
        return 0

    report_every = max(args.report_every, 1)
    for step in range(1, args.steps + 1):
        try:
            model.step()
        except Exception as exc:
            print(f"Simulation error at step {step}: {exc}")
            if args.debug_traceback:
                traceback.print_exc()
            return 1

        if args.verbose and (step % report_every == 0 or step == args.steps):
            _print_counts(step=step, model=model)

    final_counts = _count_wastes(model)
    print("Simulation finished.")
    print(
        f"Final wastes: green={final_counts['green']} "
        f"yellow={final_counts['yellow']} red={final_counts['red']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
