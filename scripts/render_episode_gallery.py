"""Render a selected set of paired reference videos and a local comparison gallery."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    p = argparse.ArgumentParser()
    for name in ['dataset', 'raw', 'mano', 'selection', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--wait-for-retarget', action='store_true')
    args = p.parse_args()
    rows = json.loads(args.selection.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    def render(row):
        folder = args.output / f"task-{row['number']:02d}"
        folder.mkdir(exist_ok=True)
        if args.wait_for_retarget:
            deadline = time.monotonic() + 7200
            path = args.dataset / 'retargeted' / f"{row['clip']}.npz"
            while True:
                try:
                    with zipfile.ZipFile(path) as archive:
                        if archive.testzip() is None:
                            break
                except (FileNotFoundError, zipfile.BadZipFile, EOFError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(path)
                time.sleep(5)
        manifest_path = folder / 'render_manifest.json'
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if previous.get('clip') != row['clip'] or previous.get('context_objects', []) != row.get('context_objects', []):
            with (folder / 'render.log').open('w') as log:
                context_args = [v for oid in row.get('context_objects', []) for v in ['--context-object', oid]]
                subprocess.run([sys.executable, str(Path(__file__).with_name('render_episode_pair.py')),
                    '--dataset', str(args.dataset), '--raw', str(args.raw), '--mano', str(args.mano),
                    '--clip', row['clip'], '--title', row['seg_desc'], '--output', str(folder), *context_args],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Rendered task {row['number']}: {row['seg_desc']}", flush=True)
        return {**row, **json.loads((folder / 'render_manifest.json').read_text())}
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ready_first = sorted(rows, key=lambda row: not (
            args.dataset / 'retargeted' / f"{row['clip']}.npz").exists())
        futures = [ex.submit(render, row) for row in ready_first]
        for future in as_completed(futures):
            results.append(future.result())
            write_gallery(sorted(results, key=lambda row: row['number']), args.output)


def write_gallery(results, output):
    (output / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    cards = []
    for row in results:
        num = row['number']
        videos = ''.join(f'<div><h3>{label}</h3><video controls preload="metadata" poster="task-{num:02d}/{mode}_0000.png" src="task-{num:02d}/{mode}_object.mp4"></video><a href="task-{num:02d}/{mode}_object.mp4" download>下载视频</a></div>'
                         for mode, label in [('human', 'Human · MANO 双手 + 物体'), ('robot', 'Robot · XHand 双手 + 物体')])
        cards.append(f'<section><h2>{num:02d} · {html.escape(row["seg_desc_zh"])}</h2><p>{row["duration_s"]:.2f} 秒 · 30 fps · 同一时间段、相同视角</p><button onclick="pair(this)">从头同步播放两段</button><div class="pair">{videos}</div><details><summary>数据标识</summary><pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></section>')
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Dex-track · 8 个任务 / 16 个视频</title><style>
body{background:#101722;color:#e3eaf4;font:16px system-ui;margin:32px auto;max-width:1500px;padding:0 24px}h1{font-size:32px}p{color:#aebed0}section{background:#1b2636;padding:24px;margin:26px 0;border-radius:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%;border-radius:8px}button{padding:10px 16px;border:0;border-radius:6px;background:#a9d6f4;cursor:pointer}a{color:#9bd4ff}pre{white-space:pre-wrap;font-size:12px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>操作任务 · Human / Robot 对照</h1><p>MANO 双手与物体参考轨迹，对照 XHand retarget 运动学回放。Retarget 在训练中仅用于 reset，不用于 observation、reward 或 termination。这里不是训练后的 policy rollout；运动学回放不能证明物理碰撞稳定。</p>'''
    document += ''.join(cards) + '''<script>function pair(b){const vs=[...b.parentElement.querySelectorAll('video')];vs.forEach(v=>{v.pause();v.currentTime=0});Promise.all(vs.map(v=>v.play())).catch(console.error)}</script></html>'''
    (output / 'index.html').write_text(document)
    print(output / 'index.html', flush=True)


if __name__ == '__main__':
    main()
