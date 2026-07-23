"""Headless generation from the command line.

Exists for two reasons: batch runs without a display, and as the standing proof that
every feature works without Qt. If this file ever needs a Qt import, the layering has
been broken.

    python -m tpms.cli --shape box --size 60 --cell 8 --wall 1.2 -o out.stl
    python -m tpms.cli --input part.step --pattern schwarz_d --resolution 256 -o out.stl
    python -m tpms.cli --settings saved.json -o out.stl
    python -m tpms.cli --list
"""

from __future__ import annotations

import argparse
import sys
import time

from tpms import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpms.cli",
        description="Generate a TPMS lattice without the GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_argument_group("source")
    source.add_argument("--input", "-i", metavar="FILE",
                        help="geometry to fill (STL, OBJ, PLY, 3MF, STEP, IGES)")
    source.add_argument("--shape", default="box",
                        help="base shape when no --input is given (default: box)")
    source.add_argument("--size", type=float, default=60.0,
                        help="nominal size of the base shape in mm (default: 60)")

    lattice = parser.add_argument_group("lattice")
    lattice.add_argument("--pattern", "-p", default="gyroid",
                         help="gyroid, schwarz_p, schwarz_d or neovius")
    lattice.add_argument("--mode", "-m", default="sheet",
                         choices=("sheet", "solid", "inverse"))
    lattice.add_argument("--cell", "-c", type=float, default=8.0,
                         help="unit cell size in mm (default: 8)")
    lattice.add_argument("--wall", "-w", type=float, default=1.2,
                         help="wall thickness in mm (default: 1.2)")
    lattice.add_argument("--fraction", type=float, default=0.5,
                         help="solid fraction for the network modes (default: 0.5)")
    lattice.add_argument("--skin", type=float, default=0.0,
                         help="solid skin thickness in mm (default: 0, none)")

    grade = parser.add_argument_group("grading")
    grade.add_argument("--grading", "-g", default="uniform",
                       help="uniform, axial, radial, surface_distance or expression")
    grade.add_argument("--cell-end", type=float,
                       help="cell size at the far end, for graded modes")
    grade.add_argument("--axis", type=int, default=2, choices=(0, 1, 2),
                       help="grading axis: 0=X, 1=Y, 2=Z (default: 2)")
    grade.add_argument("--cell-expression",
                       help="cell size as f(x,y,z), for --grading expression")
    grade.add_argument("--wall-expression",
                       help="wall thickness as f(x,y,z), for --grading expression")

    quality = parser.add_argument_group("quality")
    quality.add_argument("--resolution", "-r", type=int, default=192,
                         help="samples along the longest axis (default: 192, max 512)")
    quality.add_argument("--domain-resolution", type=int, default=160,
                         help="SDF resolution for imported geometry (default: 160)")
    quality.add_argument("--slab", type=int, default=16,
                         help="z-planes held in memory at once (default: 16)")

    parser.add_argument("--output", "-o", metavar="FILE",
                        help="where to write the result (.stl, .obj, .ply, .3mf, ...)")
    parser.add_argument("--settings", metavar="FILE",
                        help="load parameters from a saved JSON file")
    parser.add_argument("--save-settings", metavar="FILE",
                        help="write the resolved parameters to JSON")
    parser.add_argument("--list", action="store_true",
                        help="list available patterns, gradings, shapes and formats")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--version", action="version", version=f"TPMS Builder {__version__}")

    return parser


def list_options() -> None:
    from tpms.features.grading import list_gradings
    from tpms.features.shapes import list_shapes
    from tpms.features.tpms import SolidMode, list_patterns
    from tpms.io import export_extensions, registered_loaders

    print(f"TPMS Builder {__version__}\n")

    print("Patterns:")
    for pattern in list_patterns():
        print(f"  {pattern.name:18} {pattern.label} — {pattern.description}")

    print("\nSolid modes:")
    for mode in SolidMode:
        print(f"  {mode.value:18} {mode.description}")

    print("\nGrading laws:")
    for grading in list_gradings():
        print(f"  {grading.name:18} {grading.description}")

    print("\nBase shapes:")
    for shape in list_shapes():
        print(f"  {shape.name:18} {shape.description}")

    print("\nImport formats:")
    for info in registered_loaders():
        extensions = ", ".join(f".{e}" for e in info.extensions)
        reason = info.unavailable_reason()
        state = "available" if reason is None else f"UNAVAILABLE — {reason}"
        print(f"  {info.name:18} {extensions}  [{state}]")

    print("\nExport formats:")
    print("  " + ", ".join(f".{e}" for e in export_extensions()))


def _shape_params(shape: str, size: float) -> dict:
    """Map a single --size onto whichever parameters the shape actually has."""
    half = size * 0.5
    return {
        "box": {"width": size, "depth": size, "height": size},
        "sphere": {"radius": half},
        "cylinder": {"radius": half, "height": size},
        "cone": {"radius_bottom": half, "radius_top": 0.0, "height": size},
        "torus": {"major_radius": half, "minor_radius": half * 0.35},
        "capsule": {"radius": half * 0.6, "height": size * 0.6},
    }.get(shape, {"width": size, "depth": size, "height": size})


def _grading_params(args: argparse.Namespace) -> dict:
    cell_end = args.cell_end if args.cell_end is not None else args.cell * 2.0

    if args.grading == "uniform":
        return {"cell": args.cell, "wall": args.wall}
    if args.grading == "axial":
        return {"axis": args.axis, "cell_start": args.cell, "cell_end": cell_end,
                "wall_start": args.wall, "wall_end": args.wall}
    if args.grading == "radial":
        return {"axis": None, "radius_inner": 0.0, "radius_outer": args.size * 0.5,
                "cell_inner": args.cell, "cell_outer": cell_end,
                "wall_inner": args.wall, "wall_outer": args.wall}
    if args.grading == "surface_distance":
        return {"depth": args.size * 0.25,
                "cell_surface": args.cell, "cell_core": cell_end,
                "wall_surface": args.wall, "wall_core": args.wall}
    if args.grading == "expression":
        return {
            "cell_expression": args.cell_expression or str(args.cell),
            "wall_expression": args.wall_expression or str(args.wall),
        }
    return {"cell": args.cell, "wall": args.wall}


def build_params(args: argparse.Namespace):
    from tpms.features.generate import GenerationParams, SourceType

    if args.settings:
        params = GenerationParams.load(args.settings)
    else:
        params = GenerationParams()

        if args.input:
            params.source_type = SourceType.FILE
            params.source_path = args.input
        else:
            params.source_type = SourceType.SHAPE
            params.shape_name = args.shape
            params.shape_params = _shape_params(args.shape, args.size)

        params.pattern = args.pattern
        params.mode = args.mode
        params.volume_fraction = args.fraction
        params.skin_thickness = args.skin
        params.grading = args.grading
        params.grading_params = _grading_params(args)

    params.resolution = args.resolution
    params.domain_resolution = args.domain_resolution
    params.slab_depth = args.slab

    return params


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_options()
        return 0

    try:
        params = build_params(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems = params.validate()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    if args.save_settings:
        params.save(args.save_settings)
        if not args.quiet:
            print(f"Wrote settings to {args.save_settings}")

    if not args.output and not args.save_settings:
        print("error: nothing to do — pass --output or --save-settings",
              file=sys.stderr)
        return 2

    if not args.output:
        return 0

    from tpms.features.generate import generate

    last = [-1.0]

    def progress(fraction: float, message: str) -> bool:
        if args.quiet:
            return True
        # Only redraw on a whole-percent change, or the terminal does more work than
        # the mesher.
        percent = int(fraction * 100)
        if percent != last[0]:
            last[0] = percent
            print(f"\r  {percent:3d}%  {message:<44}", end="", flush=True)
        return True

    started = time.perf_counter()

    try:
        result = generate(params, progress=progress)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print()

    if result.mesh.is_empty:
        print("error: the result was empty.", file=sys.stderr)
        for warning in result.warnings:
            print(f"  {warning}", file=sys.stderr)
        return 1

    from tpms.io import export_mesh

    try:
        written = export_mesh(result.mesh, args.output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        import os

        size = os.path.getsize(written) / 1e6
        print(f"\n{result.summary()}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        print(f"\nWrote {written} ({size:.1f} MB) in {time.perf_counter() - started:.1f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
