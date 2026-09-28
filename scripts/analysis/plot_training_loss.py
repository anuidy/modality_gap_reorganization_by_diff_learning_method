"""Plot six completed CLIP runs from copied, hash-verified training logs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_retrieval_trajectories import BRANCHES


def trailing_mean(values, window=50):
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (sums[end] - sums[start]) / (end - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    for source in json.loads((root / 'sources.json').read_text()):
        assert hashlib.sha256((root / source['local']).read_bytes()).hexdigest() == source['sha256']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'path', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    detail, panels = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    validation, valax = plt.subplots(figsize=(10, 6))
    summaries = []
    for index, (branch, label, color, marker) in enumerate(BRANCHES):
        folder = root / 'source_logs' / branch
        manifest = json.loads((folder / 'run_manifest.json').read_text())
        rows = [json.loads(line) for line in (folder / 'train_metrics.jsonl').read_text().splitlines()]
        config = manifest['config']
        assert config['model_name'] == 'clip' and config['seed'] == 42
        assert config['branch'] == branch and manifest['status'] == 'complete'
        assert manifest['completed_steps'] == config['max_steps'] == 15003
        steps = np.array([row['completed_steps'] for row in rows])
        expected = np.array([1, 2, 3] + list(range(10, 15004, 10)))
        assert np.array_equal(steps, expected)
        loss = np.array([row['total_loss'] for row in rows])
        assert np.isfinite(loss).all() and (loss >= 0).all()
        assert all(row['total_loss'] == row['metrics']['loss'] for row in rows)
        x = steps / config['max_steps'] * 100
        mean = trailing_mean(loss)
        assert np.allclose(mean, [loss[max(0, i - 49):i + 1].mean() for i in range(len(loss))])
        for ax in axes:
            ax.plot(x, mean, color=color, label=label, linewidth=1.6,
                    marker=marker, markevery=150, markersize=4)
        ax = panels.flat[index]
        ax.plot(x, loss, color=color, alpha=.25, linewidth=.65, label='Logged batch loss')
        ax.plot(x, mean, color=color, linewidth=1.7, label='Trailing 50-log mean')
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Training total loss')
        ax.set_yscale('symlog', linthresh=.01)
        ax.grid(alpha=.18)
        ax.set_xlim(0, 100)
        vals = [json.loads(line) for line in (folder / 'validation_metrics.jsonl').read_text().splitlines()]
        assert [v['completed_steps'] for v in vals] == [3001, 6001, 9002, 12002, 15003]
        assert all(v['sample_count'] == 7992 and v['policy']['common_exam'] == 'I<->T' for v in vals)
        valy = [v['metrics']['common/I<->T/loss'] for v in vals]
        assert np.isfinite(valy).all()
        valax.plot([v['completed_steps'] / 15003 * 100 for v in vals], valy,
                   label=label, color=color, marker=marker, linewidth=1.7)
        summaries.append({'branch': branch, 'records': len(rows), 'last_logged_step': int(steps[-1]),
                          'completed_steps': 15003, 'first_100_logged_mean': float(loss[:100].mean()),
                          'last_100_logged_mean': float(loss[-100:].mean()),
                          'max_logged_loss': float(loss.max()), 'last_logged_loss': float(loss[-1]),
                          'final_common_validation_loss': valy[-1],
                          'final_learning_rate_logged': rows[-1]['learning_rate']})
    axes[0].set_title('Full training range')
    axes[1].set_title('Detail: loss 0 to 0.30 (larger values clipped)')
    axes[1].set_ylim(0, .30)
    axes[0].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel('Training progress (%)')
        ax.set_ylabel('Training total loss | trailing 50-log mean')
        ax.set_xlim(0, 100)
        ax.grid(alpha=.2)
    fig.suptitle('CLIP | six branches | seed 42', fontsize=18, fontweight='bold', y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .067), ncol=3, frameon=False)
    fig.text(.5, .017, '50 logged observations (~500 steps); early windows use available points. Different objectives are not a common score.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, .94))
    detail.suptitle('CLIP training loss | raw logged values + trailing mean', fontsize=18, fontweight='bold')
    for ax in panels[1]: ax.set_xlabel('Training progress (%)')
    detail.legend(*panels.flat[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .027), ncol=2, frameon=False)
    detail.text(.5, .012, 'Separate y-ranges; symmetric-log scale (linear below 0.01). Raw spikes are retained.', ha='center', fontsize=9)
    detail.tight_layout(rect=(0, .08, 1, .95))
    valax.set_title('Common I-T validation loss | CLIP | seed 42', fontsize=16, fontweight='bold')
    valax.set_xlabel('Training progress (%)'); valax.set_ylabel('Common bidirectional I-T loss')
    valax.set_xticks([20, 40, 60, 80, 100]); valax.grid(alpha=.2)
    valax.legend(ncol=2, frameon=False)
    validation.text(.5, .025, '7,992 validation pairs per point; same I-T objective; no smoothing or checkpoint selection.', ha='center', fontsize=9)
    validation.tight_layout(rect=(0, .06, 1, 1))
    for figure, name in [(fig, 'training_loss_comparison'), (detail, 'training_loss_per_branch'), (validation, 'common_validation_loss')]:
        figure.savefig(root / (name + '.png'), dpi=180)
        figure.savefig(root / (name + '.svg'))
        plt.close(figure)
    (root / 'loss_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    (root / 'README.md').write_text(
        '# CLIP 六分支训练损失\n\n'
        '范围与上一份检索表一致：seed=42，六个分支，每个完成15003步。\n'
        '日志保存第1/2/3步及每10步的单批次loss，共1503条/分支；不是每10步平均。'
        '最后日志是15000步，不把它冒充15003步的loss，也不补造最后3步。\n\n'
        'comparison为六条曲线：过去50条日志的算术平均，约500步窗口，开头用可用记录。'
        '右图只展示0–0.30；per_branch保留所有原始日志尖峰，纵轴采用symlog，0.01以下线性。'
        'common_validation_loss额外展示共同I↔T验证目标，共7992条/点。\n\n'
        '不同训练分支的监督方向、候选池和FN屏蔽不同，训练loss绝对值不能直接用来排名。'
        '平滑仅用于展示，不能用其排除瞬时异常或认定没有表征坍塌。\n\n'
        '检查点只读核对见checkpoint_inspection.json；六个100%点均为trajectory_model，'
        '没有optimizer/rng/stream。现有--resume不接受这种格式。'
        '可以增设权重初始化入口进行第二阶段训练，但需要新建优化器、明确学习率计划和数据轮次，'
        '不能称为精确恢复，也不等价于从M0连续训练更长的原计划。\n', encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
