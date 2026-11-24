"""
超参数调优结果可视化脚本 (使用原始绘图逻辑)

此脚本复用了 _plot_tuning_results 函数中的核心绘图代码，
以确保可视化结果与原始流程完全一致。

使用方法:
    python visualize_tuning_results_with_original_logic.py --csv_path /path/to/tune_results.csv

或者直接修改脚本底部的CSV路径运行
"""

import argparse
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置
import matplotlib.pyplot as plt

def format_three_decimals(x, pos):
    """定义格式化函数：保留三位小数"""
    if np.isclose(x, 0):
        return '0.000'
    elif abs(x) < 0.001:
        # 小于 0.001 的值用科学计数法表示，但保留三位有效数字
        return f'{x:.3e}'
    else:
        # 其他情况保留三位小数
        return f'{x:.3f}'

def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'text.usetex': False,
        'savefig.dpi': 300,
        'figure.dpi': 100,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'patch.linewidth': 0.5,
    })

class DummyTrainerInstance:
    """一个模拟的 trainer_instance，用于打印日志"""
    def print_to_log_file(self, msg):
        print(f"[Log] {msg}")

def visualize_tuning_results(csv_path: str, output_dir: str = None, show: bool = False):
    """
    可视化超参数调优结果 (使用原始绘图逻辑)

    Args:
        csv_path: tune_results.csv 文件路径
        output_dir: 输出目录（默认与CSV同目录）
        show: 是否显示图表 (注意: 原始代码不会显示，这里提供选项)
    """
    # 使用模拟的trainer实例
    trainer_instance = DummyTrainerInstance()

    # 设置样式
    setup_plot_style()
    formatter_3dec = FuncFormatter(format_three_decimals)

    # 读取数据
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    trainer_instance.print_to_log_file(f"成功读取CSV文件: {csv_path}")
    trainer_instance.print_to_log_file(f"总记录数: {len(df)}, 列名: {list(df.columns)}")

    # 设置输出目录
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- 开始复用原始代码的绘图逻辑 ---

    # 准备数据
    df_trials = df # 直接使用读取的DataFrame作为df_trials

    # 检查并报告fitness值的分布
    if 'fitness' in df_trials.columns:
        trainer_instance.print_to_log_file(f"Fitness statistics: min={df_trials['fitness'].min():.4f}, "
                                           f"max={df_trials['fitness'].max():.4f}, "
                                           f"mean={df_trials['fitness'].mean():.4f}")

    # 绘制参数散点图 (复用原始逻辑)
    if df_trials is not None and not df_trials.empty:
        param_keys = [col for col in df_trials.columns if col not in ['fitness', 'generation', 'individual_id']]
        fitness_values = df_trials['fitness'].values

        # 检查并报告fitness值的分布
        valid_fitness_mask = (fitness_values != 0) & (~np.isnan(fitness_values)) & (~np.isinf(fitness_values))
        if not np.any(valid_fitness_mask):
            trainer_instance.print_to_log_file("WARNING: All fitness values are 0 or invalid. "
                                               "This may indicate training failures during hyperparameter tuning.")

        if param_keys and len(fitness_values) > 0:
            # 分离数值参数和分类参数
            numeric_params = []
            categorical_params = []

            for key in param_keys:
                param_values = df_trials[key].values
                try:
                    numeric_values = pd.to_numeric(param_values, errors='coerce')
                    if not numeric_values.isna().all():
                        numeric_params.append(key)
                    else:
                        categorical_params.append(key)
                except:
                    categorical_params.append(key)

            all_params = numeric_params + categorical_params
            num_keys = len(all_params)
            n_cols_plot = min(3, num_keys)
            n_rows_plot = (num_keys + n_cols_plot - 1) // n_cols_plot

            fig_scatter, axes_scatter = plt.subplots(n_rows_plot, n_cols_plot,
                                                     figsize=(7 * n_cols_plot, 5.5 * n_rows_plot),
                                                     squeeze=False)
            if n_rows_plot == 1 and n_cols_plot == 1:
                axes_scatter = np.array([[axes_scatter]])
            elif n_rows_plot == 1:
                axes_scatter = axes_scatter.reshape(1, -1)
            elif n_cols_plot == 1:
                axes_scatter = axes_scatter.reshape(-1, 1)

            axes_scatter_flat = axes_scatter.ravel()

            # 找到最佳fitness的索引（忽略无效值）
            valid_indices = np.where(valid_fitness_mask)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(fitness_values[valid_indices])]
            else:
                best_idx = np.argmax(fitness_values)

            for i, key in enumerate(all_params):
                param_values = df_trials[key].values
                ax = axes_scatter_flat[i]

                if len(param_values) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue

                generations = df_trials['generation'].values

                if key in categorical_params:
                    unique_values = list(set(param_values))
                    value_to_int = {v: idx for idx, v in enumerate(unique_values)}
                    numeric_param_values = [value_to_int[v] for v in param_values]

                    if len(generations) > 0 and np.max(generations) > np.min(generations):
                        sc = ax.scatter(numeric_param_values, fitness_values, c=generations, cmap='viridis',
                                       alpha=0.7, edgecolors='black', s=25, linewidth=0.5)
                        try:
                            cbar = plt.colorbar(sc, ax=ax)
                            cbar.set_label('Generation', fontsize=8)
                            cbar.ax.tick_params(labelsize=8)
                        except Exception as e:
                            trainer_instance.print_to_log_file(f"Warning: Could not create colorbar for {key}: {e}")
                    else:
                        ax.scatter(numeric_param_values, fitness_values, alpha=0.7, s=25,
                                  edgecolors='black', linewidth=0.5, c='blue')

                    ax.set_xticks(range(len(unique_values)))
                    ax.set_xticklabels([str(v) for v in unique_values], rotation=45)

                    if 0 <= best_idx < len(param_values):
                        best_param_value = param_values[best_idx]
                        best_numeric_value = value_to_int[best_param_value]
                        ax.plot(best_numeric_value, fitness_values[best_idx], 'r*', markersize=18,
                               label=f'Best ({best_param_value})', markeredgewidth=1.2)

                    # 分类参数：Y轴（fitness）使用三位小数格式
                    ax.yaxis.set_major_formatter(formatter_3dec)

                else:
                    try:
                        numeric_param_values = pd.to_numeric(param_values, errors='coerce')
                        valid_mask = ~numeric_param_values.isna()

                        if valid_mask.sum() == 0:
                            ax.text(0.5, 0.5, 'No valid numeric data', ha='center', va='center', transform=ax.transAxes)
                            continue

                        valid_param_values = numeric_param_values[valid_mask].values
                        valid_fitness_values = fitness_values[valid_mask]
                        valid_generations = generations[valid_mask]

                        if len(valid_generations) > 0 and np.max(valid_generations) > np.min(valid_generations):
                            sc = ax.scatter(valid_param_values, valid_fitness_values, c=valid_generations,
                                           cmap='viridis', alpha=0.7, edgecolors='black', s=25, linewidth=0.5)
                            try:
                                cbar = plt.colorbar(sc, ax=ax)
                                cbar.set_label('Generation', fontsize=8)
                                cbar.ax.tick_params(labelsize=8)
                            except Exception as e:
                                trainer_instance.print_to_log_file(f"Warning: Could not create colorbar for {key}: {e}")
                        else:
                            ax.scatter(valid_param_values, valid_fitness_values, alpha=0.7, s=25,
                                      edgecolors='black', linewidth=0.5, c='blue')

                        # 标记最佳点
                        if 0 <= best_idx < len(param_values):
                            try:
                                if hasattr(valid_mask, 'iloc'):
                                    is_valid = valid_mask.iloc[best_idx]
                                else:
                                    is_valid = valid_mask[best_idx]

                                if is_valid:
                                    if hasattr(numeric_param_values, 'iloc'):
                                        best_param_val = numeric_param_values.iloc[best_idx]
                                    else:
                                        best_param_val = numeric_param_values[best_idx]
                                    ax.plot(best_param_val, fitness_values[best_idx], 'r*', markersize=18,
                                           label=f'Best ({best_param_val:.3f})', markeredgewidth=1.2)
                            except Exception as e:
                                trainer_instance.print_to_log_file(f"Warning: Could not mark best point for {key}: {e}")

                        # 数值参数：X轴和Y轴都使用三位小数格式
                        ax.xaxis.set_major_formatter(formatter_3dec)
                        ax.yaxis.set_major_formatter(formatter_3dec)

                    except Exception as e:
                        trainer_instance.print_to_log_file(f"Error processing numeric parameter {key}: {e}")
                        continue

                ax.set_xlabel(key, fontsize=12, fontweight='medium')
                ax.set_ylabel('Fitness (Dice Score)', fontsize=12, fontweight='medium')
                ax.set_title(f'{key} vs Fitness', fontsize=14, fontweight='bold', pad=15)
                ax.legend(fontsize=10, framealpha=0.9, edgecolor='gray')
                ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
                ax.tick_params(axis='both', which='major', labelsize=10)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)

            for i in range(num_keys, len(axes_scatter_flat)):
                fig_scatter.delaxes(axes_scatter_flat[i])

            plt.tight_layout(pad=3.0)
            scatter_plot_path = output_dir / 'tune_scatter_plots_visualized.png' # 保持原始文件名
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            trainer_instance.print_to_log_file(f"Saved scatter plots to {scatter_plot_path}")

            if show:
                plt.show()
            plt.close(fig_scatter) # 确保关闭

    # 绘制适应度进化图 (复用原始逻辑)
    fig_fitness, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11))

    # 上图：每代最佳适应度
    if 'generation' in df_trials.columns:
        gen_groups = df_trials.groupby('generation')['fitness']
        best_fitness_per_gen_raw = gen_groups.max().values
        generations_list = sorted(df_trials['generation'].unique())

        # 过滤无效值 (复用原始逻辑)
        valid_fitness = [f if (f is not None and not np.isnan(f) and not np.isinf(f)) else 0.0
                         for f in best_fitness_per_gen_raw]

        ax1.plot(generations_list, valid_fitness, marker='o', linestyle='-', color='#2E86AB',
                 linewidth=2.5, markersize=7, markerfacecolor='#2E86AB', markeredgecolor='white',
                 markeredgewidth=1.5, label='Best Fitness per Generation')

        if len(valid_fitness) >= 3:
            try:
                smoothed_best_fitness = gaussian_filter1d(valid_fitness, sigma=1)
                ax1.plot(generations_list, smoothed_best_fitness, '--', linewidth=2.5, color='#A23B72',
                         alpha=0.8, label='Smoothed Best Fitness')
            except Exception as e:
                trainer_instance.print_to_log_file(f"Warning: Could not create smoothed line: {e}")

        ax1.set_title('Best Fitness Evolution - Medical Image Segmentation',
                      fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Generation', fontsize=12, fontweight='medium')
        ax1.set_ylabel('Dice Score', fontsize=12, fontweight='medium')
        ax1.legend(fontsize=11, framealpha=0.9, edgecolor='gray', loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # --- 🔧 修正：先设置定位器，再设置格式化器 (来自原始代码) ---
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.xaxis.set_major_formatter(formatter_3dec)  # X轴格式化（虽然这里是整数，但以防万一）
        ax1.yaxis.set_major_formatter(formatter_3dec)  # Y轴格式化（关键！）
        # ------------------------------------------------

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1.2)
        ax1.spines['bottom'].set_linewidth(1.2)
    else:
        ax1.text(0.5, 0.5, 'No generation column available', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, style='italic')

    # 下图：所有试验的适应度分布
    if df_trials is not None and not df_trials.empty and len(df_trials) > 0:
        trial_numbers = range(len(df_trials))
        trials_fitness = df_trials['fitness'].values

        # 过滤无效值用于显示 (复用原始逻辑)
        valid_trials_fitness = np.array([f if (not np.isnan(f) and not np.isinf(f)) else 0.0
                                         for f in trials_fitness])

        ax2.scatter(trial_numbers, valid_trials_fitness, alpha=0.6, s=20, c='#F18F01',
                    edgecolors='white', linewidth=0.5, label='Individual Evaluations')

        if len(valid_trials_fitness) >= 5:
            try:
                smoothed_trials_fitness = gaussian_filter1d(valid_trials_fitness,
                                                            sigma=max(1, len(valid_trials_fitness) // 20))
                ax2.plot(trial_numbers, smoothed_trials_fitness, '-', linewidth=2.5, color='#C73E1D',
                         label='Smoothed Trend')
            except Exception as e:
                trainer_instance.print_to_log_file(f"Warning: Could not create smoothed trend: {e}")

        ax2.set_title('All Evaluations Fitness Distribution',
                      fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Evaluation Number', fontsize=12, fontweight='medium')
        ax2.set_ylabel('Dice Score', fontsize=12, fontweight='medium')
        ax2.legend(fontsize=11, framealpha=0.9, edgecolor='gray', loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax2.tick_params(axis='both', which='major', labelsize=10)

        # --- 🔧 修正：先设置定位器，再设置格式化器 (来自原始代码) ---
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_formatter(formatter_3dec)  # X轴格式化
        ax2.yaxis.set_major_formatter(formatter_3dec)  # Y轴格式化（关键！）
        # ------------------------------------------------

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(1.2)
        ax2.spines['bottom'].set_linewidth(1.2)
    else:
        ax2.text(0.5, 0.5, 'No trial data available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, style='italic')

    plt.tight_layout(pad=3.0)
    evolution_plot_path = output_dir / 'fitness_evolution_visualized.png' # 保持原始文件名
    plt.savefig(evolution_plot_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    trainer_instance.print_to_log_file(f"Saved fitness evolution plot to {evolution_plot_path}")

    if show:
        plt.show()
    plt.close(fig_fitness) # 确保关闭

    # --- 结束复用原始代码的绘图逻辑 ---

    # 生成简单的文本报告
    if 'fitness' in df.columns:
        best_idx_overall = df['fitness'].idxmax()
        best_fitness_overall = df.loc[best_idx_overall, 'fitness']
        best_params_overall = df.loc[best_idx_overall, [col for col in df.columns if col not in ['fitness', 'generation', 'individual_id']]]

        report_lines = [
            "# nnU-Net Hyperparameter Tuning Report (Visualization)",
            "=" * 50,
            f"CSV Source: {csv_path}",
            "",
            "## Overall Best Result",
            f"Best Dice Score: {best_fitness_overall:.4f}",
            f"Index in CSV: {best_idx_overall}",
        ]

        if 'generation' in df.columns:
             best_generation = int(df.loc[best_idx_overall, 'generation'])
             report_lines.append(f"Generation: {best_generation}")

        report_lines.append("")
        report_lines.append("## Best Hyperparameters Found:")
        for key, value in best_params_overall.items():
            report_lines.append(f"  - {key}: {value:.6g}" if isinstance(value, (int, float)) else f"  - {key}: {value}")

        report_content = "\n".join(report_lines)

        report_path = output_dir / 'visualization_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        trainer_instance.print_to_log_file(f"Generated visualization report: {report_path}")

    # 重置 matplotlib 参数
    plt.rcdefaults()

    trainer_instance.print_to_log_file("\n✅ 可视化完成！")


def main():
    parser = argparse.ArgumentParser(description='可视化超参数调优结果 (使用原始绘图逻辑)')
    parser.add_argument('--csv_path', type=str,
                        default=r"\\wsl.localhost\Ubuntu\home\userc\data\nnUNet_results\Dataset654_Rib28nnUNet\nnUNetTrainerNoMirroringMedHybrid3D__nnUNetPlans__3d_fullres\fold_0\hyperparameter_tuning\tune_results.csv",
                        help='tune_results.csv 文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与CSV同目录）')
    parser.add_argument('--show', action='store_true', # 修改为 --show 才显示
                        help='显示图表 (默认不显示，仅保存)')

    args = parser.parse_args()

    visualize_tuning_results(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        show=args.show
    )


if __name__ == '__main__':
    main()