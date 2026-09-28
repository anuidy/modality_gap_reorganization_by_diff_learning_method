"""Plot verified checkpoint-level retrieval summaries without fitting or smoothing curves."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BRANCHES = [
    ('standard', 'Standard', '#374151', 'o'),
    ('fixed_2m', 'Fixed-2M', '#0072B2', 's'),
    ('fixed_3m_fn_off', 'Fixed-3M FN-off', '#009E73', '^'),
    ('fixed_3m_fn_on', 'Fixed-3M FN-on', '#D55E00', 'v'),
    ('mixed_2m', 'Mixed-2M', '#AA4499', 'D'),
    ('mixed_3m_fn_off', 'Mixed-3M FN-off', '#B8860B', 'P'),
]
POINTS = [1, 5, 20, 50, 100]
METRICS = ['R@1', 'R@5', 'R@10']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding='utf-8'))
    if not source['all_artifact_hashes_verified'] or not source['same_evaluation_view_per_task']:
        raise ValueError('Input must have complete artifact and protocol verification')
    if source['model'] != 'clip' or source['seed'] != 42 or source['points_percent'] != POINTS:
        raise ValueError('Unexpected plotting scope')
    rows = {}
    for row in source['rows']:
        key = (row['stage'], row['task'], row['branch'], row['progress_percent'])
        if key in rows: raise ValueError('Duplicate task/branch/checkpoint observation')
        rows[key] = row
    tasks = [('B1', 'coco'), ('B1', 'flickr30k')] + [('B_Local', t) for t in source['local_tasks']]
    expected = {(stage, task, branch, progress) for stage, task in tasks
                for branch, _, _, _ in BRANCHES for progress in POINTS}
    if set(rows) != expected: raise ValueError('Incomplete or unexpected six-branch coverage')
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / 'figures'; figures.mkdir(exist_ok=True)
    if any(figures.iterdir()): raise FileExistsError('Use an empty figure directory')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 12,
                         'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#9CA3AF', 'xtick.color': '#374151', 'ytick.color': '#374151',
                         'savefig.facecolor': 'white', 'svg.fonttype': 'path'})
    plot_manifest = {'model': 'clip', 'seed': 42, 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                     'collected_at': source['collected_at'], 'checkpoint_percent': POINTS,
                     'x_spacing': 'actual_numeric_progress', 'y_units': 'percent', 'y_limits': [0, 100],
                     'smoothing': False, 'confidence_intervals': False, 'figures': [], 'panels': []}
    figure_count = panel_count = series_count = point_count = 0
    for stage, task in tasks:
        directions = ['I->T', 'T->I'] if stage == 'B1' else [None]
        fig, axes = plt.subplots(len(directions), 3, figsize=(15, 8.2 if stage == 'B1' else 5.0), squeeze=False)
        title = {'coco': 'COCO Karpathy', 'flickr30k': 'Flickr30K'}.get(task, task)
        fig.suptitle(title + ('  |  B1 retrieval' if stage == 'B1' else '  |  Local retrieval'),
                     fontsize=19, fontweight='bold', x=.055, ha='left', y=.965)
        if stage == 'B1':
            m = rows[(stage, task, 'standard', 1)]['metrics']
            detail = f"{m['image_candidates']:,} images / {m['text_candidates']:,} original captions"
        else:
            m = rows[(stage, task, 'standard', 1)]['metrics']
            detail = f"{m['queries']:,} queries / {m['candidates_per_query']:,} candidates per query"
        fig.text(.055, .909 if stage == 'B1' else .885, 'CLIP ViT-L/14  |  seed 42  |  ' + detail,
                 color='#64748B', fontsize=10.5)
        for i, direction in enumerate(directions):
            for j, metric in enumerate(METRICS):
                ax = axes[i, j]; panel = {'stage': stage, 'task': task, 'direction': direction, 'metric': metric, 'series': []}
                for branch, label, color, marker in BRANCHES:
                    values = []
                    references = []
                    for progress in POINTS:
                        row = rows[(stage, task, branch, progress)]
                        measured = row['metrics'][direction] if direction else row['metrics']
                        values.append(100 * measured[metric]); references.append(row['summary_sha256'])
                    y = np.asarray(values, np.float64)
                    if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 100): raise ValueError('Invalid recall values')
                    line, = ax.plot(POINTS, y, color=color, label=label, marker=marker, linewidth=2.1,
                                    markersize=5.2, markeredgewidth=.7, markeredgecolor='white')
                    np.testing.assert_array_equal(line.get_xdata(), POINTS)
                    np.testing.assert_array_equal(line.get_ydata(), y)
                    panel['series'].append({'branch': branch, 'label': label, 'color': color,
                                            'recall_percent': values, 'source_summary_sha256': references})
                    series_count += 1; point_count += len(y)
                panel_count += 1; plot_manifest['panels'].append(panel)
                direction_title = {'I->T': 'Image → Text', 'T->I': 'Text → Image'}.get(direction)
                ax.set_title((direction_title + '  ·  ' if direction_title else '') + metric, loc='left', pad=10)
                ax.set_xlim(-2, 104); ax.set_ylim(0, 100)
                ax.set_xticks(POINTS, [f'{x}%' for x in POINTS])
                labels = ax.get_xticklabels(); labels[0].set_ha('right'); labels[1].set_ha('left')
                ax.set_yticks(np.arange(0, 101, 20))
                ax.grid(axis='y', color='#E2E8F0', linewidth=.8); ax.set_axisbelow(True)
                ax.set_xlabel('Checkpoint progress')
                if j == 0: ax.set_ylabel('Recall (%)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .021), ncol=3,
                   frameon=False, fontsize=10.5, handlelength=2.6, columnspacing=2.5, labelspacing=.8)
        fig.subplots_adjust(left=.055, right=.98, top=.855 if stage == 'B1' else .775,
                            bottom=.16 if stage == 'B1' else .265, hspace=.43, wspace=.23)
        name = ('B1_' if stage == 'B1' else 'Local_') + task
        png = figures / (name + '.png'); svg = figures / (name + '.svg')
        fig.savefig(png, dpi=180); fig.savefig(svg)
        plot_manifest['figures'].append({'stage': stage, 'task': task, 'title': title,
                                        'png': 'figures/' + png.name, 'svg': 'figures/' + svg.name,
                                        'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
                                        'svg_sha256': hashlib.sha256(svg.read_bytes()).hexdigest()})
        plt.close(fig); figure_count += 1
    if (figure_count, panel_count, series_count, point_count) != (14, 48, 288, 1440):
        raise ValueError('Unexpected chart coverage')
    plot_manifest['checks'] = {'figures': figure_count, 'panels': panel_count,
                               'lines': series_count, 'plotted_recall_values': point_count,
                               'all_points_match_source': True}
    (args.output / 'plot_manifest.json').write_text(json.dumps(plot_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    cards = []; links = []
    for f in plot_manifest['figures']:
        anchor = ('B1_' if f['stage'] == 'B1' else 'Local_') + f['task']
        links.append(f'<a href="#{html.escape(anchor)}">{html.escape(f["title"])}</a>')
        cards.append(f'<section id="{html.escape(anchor)}"><h2>{html.escape(f["title"])} · {f["stage"]}</h2>'
                     f'<p><a href="{f["png"]}" target="_blank">查看原尺寸PNG</a> · <a href="{f["svg"]}" target="_blank">矢量SVG</a></p>'
                     f'<a href="{f["png"]}" target="_blank"><img loading="lazy" src="{f["png"]}" alt="{html.escape(f["title"])} 六分支检查点轨迹"></a></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLIP 六分支 B1 / Local 评测轨迹</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f1f5f9;color:#172033}main{max-width:1500px;margin:auto;padding:32px 20px}
h1{font-size:28px;margin-bottom:10px}p{line-height:1.7}nav{display:flex;gap:9px;flex-wrap:wrap;margin:24px 0}a{color:#125e91}nav a{background:white;border:1px solid #dbe3ed;padding:7px 11px;border-radius:6px;text-decoration:none}
section{background:white;padding:18px;border:1px solid #dbe3ed;border-radius:10px;margin:24px 0}h2{font-size:20px;margin:4px 0}section p{margin:6px 0}img{width:100%;height:auto;display:block}code{background:#e2e8f0;padding:2px 5px}.note{color:#475569}
</style><main><h1>CLIP 六分支：B1 / Local 评测轨迹</h1>
<p>每个面板六条线，横轴使用真实训练进度1%、5%、20%、50%、100%，纵轴为召回率百分比。Local每任务三个面板；B1每数据集六个面板，包含双向R@1/5/10。</p>
<p class="note">固定六个已完成分支，seed（随机种子）=42；不包含M0和后续新完成分支。单seed不画误差条，仅连接五个实测点。所有图使用相同颜色及0–100%纵轴。</p>
<p><a href="verified_metrics.json">完整指标与来源JSON（含B1 mR及其他原始指标）</a> · <a href="plot_manifest.json">绘图数值与校验记录</a></p>'''
    page += '<p class="note">数据快照：' + html.escape(source['collected_at']) + '</p><nav>' + ''.join(links) + '</nav>' + ''.join(cards) + '</main></html>'
    (args.output / 'index.html').write_text(page, encoding='utf-8')
    (args.output / 'README.md').write_text('# CLIP six-branch checkpoint plots\n\nOpen index.html to browse all 14 figures. Each panel has six curves; x uses actual checkpoint progress and y is recall in percent.\n\nverified_metrics.json preserves all 420 B1/Local task summaries and source identities; plot_manifest.json records all 1440 plotted values.\n\nCOCO uses all 25,010 original captions. No M0 point, smoothing, confidence interval or cross-seed inference is included. PNG and SVG files are under figures/.\n', encoding='utf-8')
    archive = args.output / 'CLIP_six_branches_B1_Local_plots.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(args.output.rglob('*')):
            if path.is_file() and path != archive: bundle.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps({**plot_manifest['checks'], 'output': str(args.output), 'archive_bytes': archive.stat().st_size}))


if __name__ == '__main__': main()
