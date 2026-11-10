"""Regenerate excluded object/support USD caches; object conversion requires IsaacLab."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--object', action='append', default=[])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--urdf-only', action='store_true', help='Author object URDFs without Isaac; does not generate object USD')
    parser.add_argument('--supports-only', action='store_true', help='Regenerate portable support USDA without Isaac')
    args = parser.parse_args()
    root = args.dataset.resolve()
    if root.is_file():
        root = root.parent
    index = json.loads((root / 'index.json').read_text())
    if index.get('schema') != 'dex-track-motion-folders-v3':
        parser.error('Expected pose-only v3 dataset root or index.json')
    project = Path(__file__).resolve().parents[1]
    bundled = Path(__file__).parent
    support_path = project / 'dex_track/supports.py'
    support = runpy.run_path(str(support_path if support_path.is_file() else bundled / 'supports.py'))
    layouts = {}
    for row in index['motions']:
        motion = json.loads((root / row['path']).read_text())
        for items in motion.get('motion_scene_supports', []):
            directory = support['support_asset_directory'](items)
            if directory:
                layouts[directory] = items
    for directory, items in sorted(layouts.items()):
        target = root / directory / '.isaac_usd/object.usd'
        if target.exists() and not args.force:
            continue
        # Generate in isolation: never rewrite existing physical geometry.
        with tempfile.TemporaryDirectory() as temp:
            support['write_support_asset'](Path(temp), items)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix('.partial.usd')
            shutil.copyfile(Path(temp) / directory / '.isaac_usd/object.usd', partial)
            partial.replace(target)
        print(target, flush=True)
    if args.supports_only:
        return
    source = project / 'third_party/FlashDexRetarget/src/simulator/isaacsim/convert_assets.py'
    converter = runpy.run_path(str(source if source.is_file() else bundled / 'convert_assets.py'))
    objects = sorted(p for p in (root / 'objects').iterdir() if p.is_dir() and (not args.object or p.name in args.object))
    if not objects:
        parser.error('No objects matched the selection')
    app = None
    if not args.urdf_only:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True).app
    try:
        for obj in objects:
            if args.urdf_only:
                result = converter['author_object_urdf'](str(obj), str(obj / '.isaac_usd'))
                print(result, flush=True)
            else:
                converter['convert_object'](str(obj), force=args.force)
    finally:
        if app is not None:
            app.close()
    print('Object exports use original reference convex hulls, preserving the previous cache convention.')
    print('This does not add Isaac support for the current free-hand/simplified-collision training task.')


if __name__ == '__main__':
    main()
