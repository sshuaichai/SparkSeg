def _plot_tuning_results(trainer_instance, save_path: str, genetic_optimizer):
    """绘制调优结果图表 - 增强版本"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from scipy.ndimage import gaussian_filter1d
        import yaml
        from pathlib import Path
        import numpy as np
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        # 定义格式化函数：保留三位小数
        def format_three_decimals(x, pos):
            if np.isclose(x, 0):
                return '0.000'
            elif abs(x) < 0.001:
                # 小于 0.001 的值用科学计数法表示，但保留三位有效数字
                return f'{x:.3e}'
            else:
                # 其他情况保留三位小数
                return f'{x:.3f}'
        
        formatter_3dec = FuncFormatter(format_three_decimals)

        # 设置 Times New Roman 字体和 LaTeX 渲染
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

        save_path_p = Path(save_path)
        save_path_p.mkdir(parents=True, exist_ok=True)
        trainer_instance.print_to_log_file(f"Plotting tuning results to {save_path_p}")

        # 准备数据用于CSV和绘图
        all_trials_data = []
        if hasattr(genetic_optimizer, 'population_history_for_csv') and genetic_optimizer.population_history_for_csv:
            for gen_idx, gen_pop in enumerate(genetic_optimizer.population_history_for_csv):
                for ind_idx, ind in enumerate(gen_pop):
                    row = ind.params.copy()
                    # 确保fitness是有效数值
                    fitness_val = ind.fitness
                    if fitness_val is None or np.isnan(fitness_val) or np.isinf(fitness_val):
                        fitness_val = 0.0
                    row['fitness'] = float(fitness_val)
                    row['generation'] = gen_idx
                    row['individual_id'] = ind_idx
                    all_trials_data.append(row)

        df_trials = None
        if all_trials_data:
            df_trials = pd.DataFrame(all_trials_data)
            # 确保 'fitness' 是第一列
            if 'fitness' in df_trials.columns:
                cols = ['fitness', 'generation'] + [col for col in df_trials.columns if
                                                    col not in ['fitness', 'generation']]
                df_trials = df_trials[cols]
            
            # 检查并报告fitness值的分布
            trainer_instance.print_to_log_file(f"Fitness statistics: min={df_trials['fitness'].min():.4f}, "
                                               f"max={df_trials['fitness'].max():.4f}, "
                                               f"mean={df_trials['fitness'].mean():.4f}")

            df_trials.to_csv(save_path_p / 'tune_results.csv', index=False)
            trainer_instance.print_to_log_file(f"Saved tuning trials to {save_path_p / 'tune_results.csv'}")

        # 绘制参数散点图
        if df_trials is not None and not df_trials.empty:
            param_keys = [col for col in df_trials.columns if col not in ['fitness', 'generation', 'individual_id']]
            fitness_values = df_trials['fitness'].values

            # 检查fitness是否全为0或无效
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
                plt.savefig(save_path_p / 'tune_scatter_plots.png', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                plt.close(fig_scatter)
                trainer_instance.print_to_log_file(f"Saved scatter plots to {save_path_p / 'tune_scatter_plots.png'}")

        # 绘制适应度进化图
        fig_fitness, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11))

        # 上图：每代最佳适应度
        if hasattr(genetic_optimizer, 'fitness_history') and genetic_optimizer.fitness_history:
            generations = range(len(genetic_optimizer.fitness_history))
            best_fitness_per_gen = genetic_optimizer.fitness_history

            # 过滤无效值
            valid_fitness = [f if (f is not None and not np.isnan(f) and not np.isinf(f)) else 0.0
                             for f in best_fitness_per_gen]

            ax1.plot(generations, valid_fitness, marker='o', linestyle='-', color='#2E86AB',
                     linewidth=2.5, markersize=7, markerfacecolor='#2E86AB', markeredgecolor='white',
                     markeredgewidth=1.5, label='Best Fitness per Generation')

            if len(valid_fitness) >= 3:
                try:
                    smoothed_best_fitness = gaussian_filter1d(valid_fitness, sigma=1)
                    ax1.plot(generations, smoothed_best_fitness, '--', linewidth=2.5, color='#A23B72',
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

            # --- 🔧 修正：先设置定位器，再设置格式化器 ---
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.xaxis.set_major_formatter(formatter_3dec)  # X轴格式化（虽然这里是整数，但以防万一）
            ax1.yaxis.set_major_formatter(formatter_3dec)  # Y轴格式化（关键！）
            # ------------------------------------------------

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_linewidth(1.2)
            ax1.spines['bottom'].set_linewidth(1.2)
        else:
            ax1.text(0.5, 0.5, 'No fitness history available', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=12, style='italic')

        # 下图：所有试验的适应度分布
        if df_trials is not None and not df_trials.empty and len(df_trials) > 0:
            trial_numbers = range(len(df_trials))
            trials_fitness = df_trials['fitness'].values

            # 过滤无效值用于显示
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

            # --- 🔧 修正：先设置定位器，再设置格式化器 ---
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
        plt.savefig(save_path_p / 'fitness_evolution.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        # plt.show() # 确保这里也没有 plt.show()
        plt.close(fig_fitness)
        trainer_instance.print_to_log_file(f"Saved fitness evolution plot to {save_path_p / 'fitness_evolution.png'}")

        # 保存最佳超参数和详细信息
        if hasattr(genetic_optimizer, 'best_individual') and genetic_optimizer.best_individual:
            best_params_to_save = {
                'best_hyperparameters': genetic_optimizer.best_individual.params,
                'best_fitness_dice_score': float(genetic_optimizer.best_individual.fitness),
                'optimization_summary': {
                    'total_evaluations': len(all_trials_data) if all_trials_data else 0,
                    'generations': len(genetic_optimizer.fitness_history) if hasattr(genetic_optimizer,
                                                                                     'fitness_history') else 0,
                    'improvement': float(
                        genetic_optimizer.fitness_history[-1] - genetic_optimizer.fitness_history[0]) if hasattr(
                        genetic_optimizer, 'fitness_history') and len(genetic_optimizer.fitness_history) > 1 else 0.0,
                    'search_space': genetic_optimizer.search_space if hasattr(genetic_optimizer, 'search_space') else {}
                }
            }

            with open(save_path_p / 'best_hyperparameters.yaml', 'w') as f:
                yaml.dump(best_params_to_save, f, sort_keys=False, default_flow_style=False, indent=2)
            trainer_instance.print_to_log_file(
                f"Saved best hyperparameters to {save_path_p / 'best_hyperparameters.yaml'}")

        # 生成调优报告
        _generate_tuning_report(trainer_instance, save_path_p, genetic_optimizer, df_trials)

        # 重置 matplotlib 参数
        plt.rcdefaults()

    except ImportError as e:
        trainer_instance.print_to_log_file(f"Failed to import required libraries for plotting: {e}")
    except Exception as e:
        trainer_instance.print_to_log_file(f"Error during _plot_tuning_results: {e}")
        import traceback
        trainer_instance.print_to_log_file(traceback.format_exc())


def _generate_tuning_report(trainer_instance, save_path_p, genetic_optimizer, df_trials):
    """生成详细的调优报告"""
    try:
        report_lines = [
            "# nnU-Net Hyperparameter Tuning Report",
            "=" * 50,
            f"Dataset: {getattr(trainer_instance, 'dataset_json', {}).get('name', 'Unknown')}",
            f"Configuration: {getattr(trainer_instance, 'configuration', 'Unknown')}",
            f"Fold: {getattr(trainer_instance, 'fold', 'Unknown')}",
            "",
            "## Optimization Results",
        ]

        if hasattr(genetic_optimizer, 'best_individual') and genetic_optimizer.best_individual:
            report_lines.append(f"Best Dice Score: {genetic_optimizer.best_individual.fitness:.4f}")
        else:
            report_lines.append("Best Dice Score: N/A")

        report_lines.extend([
            f"Total Evaluations: {len(df_trials) if df_trials is not None else 0}",
            f"Generations: {len(genetic_optimizer.fitness_history) if hasattr(genetic_optimizer, 'fitness_history') else 0}",
            "",
            "## Best Hyperparameters",
        ])

        if hasattr(genetic_optimizer, 'best_individual') and genetic_optimizer.best_individual:
            for key, value in genetic_optimizer.best_individual.params.items():
                report_lines.append(f"{key}: {value:.6g}" if isinstance(value, (int, float)) else f"{key}: {value}")
        else:
            report_lines.append("No best hyperparameters available")

        if df_trials is not None and not df_trials.empty:
            report_lines.extend([
                "",
                "## Parameter Statistics",
                f"Mean Dice Score: {df_trials['fitness'].mean():.4f}",
                f"Std Dice Score: {df_trials['fitness'].std():.4f}",
                f"Min Dice Score: {df_trials['fitness'].min():.4f}",
                f"Max Dice Score: {df_trials['fitness'].max():.4f}",
            ])

        report_content = "\n".join(report_lines)

        with open(save_path_p / 'tuning_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)

        trainer_instance.print_to_log_file(f"Generated tuning report: {save_path_p / 'tuning_report.txt'}")

    except Exception as e:
        trainer_instance.print_to_log_file(f"Error generating tuning report: {e}")
        import traceback
        trainer_instance.print_to_log_file(traceback.format_exc())