import random
import numpy as np
from typing import Dict, Tuple, List, Any, Callable
import torch
import copy
from dataclasses import dataclass
from pathlib import Path
import json
import os


@dataclass
class Individual:
    """遗传算法个体"""
    params: Dict[str, Any]
    fitness: float = float('-inf')
    generation: int = 0


class GeneticOptimizer:
    """遗传算法超参数优化器"""

    def __init__(
            self,
            search_space: Dict[str, Tuple[float, float]],
            population_size: int = 20,
            generations: int = 10,
            mutation_rate: float = 0.1,
            crossover_rate: float = 0.8,
            elite_ratio: float = 0.2,
            random_seed: int = 42,
            on_generation_end: Callable = None,  # 新增：每代结束回调
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = int(population_size * elite_ratio)
        self.random_seed = random_seed
        self.on_generation_end = on_generation_end  # 回调函数

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.population = []
        self.best_individual = None
        self.fitness_history = []
        self.population_history_for_csv = []  # 新增：用于存储每一代的种群历史
        
        # 分析搜索空间类型
        self.discrete_params = {}
        self.continuous_params = {}
        self.integer_list_params = {}  # 新增：整数列表参数
        self.log_scale_params = {'initial_lr', 'weight_decay', 'eps', 'kl_weight'}  # 对数尺度参数
        
        for param_name, param_range in search_space.items():
            if isinstance(param_range, list):
                # 检查是否是整数列表
                if all(isinstance(x, int) for x in param_range):
                    self.integer_list_params[param_name] = param_range
                else:
                    # 离散参数（如优化器类型、布尔值）
                    self.discrete_params[param_name] = param_range
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # 连续参数
                self.continuous_params[param_name] = param_range

    def _generate_random_params(self) -> Dict[str, Any]:
        """生成随机参数 - 支持离散和连续参数"""
        params = {}
        
        # 处理离散参数
        for param_name, choices in self.discrete_params.items():
            params[param_name] = random.choice(choices)
        
        # 处理整数列表参数
        for param_name, choices in self.integer_list_params.items():
            params[param_name] = int(random.choice(choices))
        
        # 处理连续参数
        for param_name, (min_val, max_val) in self.continuous_params.items():
            if param_name in self.log_scale_params:
                # 对数空间搜索
                params[param_name] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
            else:
                # 线性空间搜索
                params[param_name] = np.random.uniform(min_val, max_val)
        
        return params

    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            params = self._generate_random_params()
            individual = Individual(params=params, generation=0)
            self.population.append(individual)

    def _mutate(self, individual: Individual) -> Individual:
        """变异操作 - 支持离散和连续参数"""
        mutated_params = copy.deepcopy(individual.params)

        for param_name, value in mutated_params.items():
            if random.random() < self.mutation_rate:
                if param_name in self.discrete_params:
                    # 离散参数变异：随机选择新值
                    choices = self.discrete_params[param_name]
                    mutated_params[param_name] = random.choice(choices)
                
                elif param_name in self.integer_list_params:
                    # 整数列表参数变异：随机选择新值
                    choices = self.integer_list_params[param_name]
                    mutated_params[param_name] = int(random.choice(choices))
                    
                elif param_name in self.continuous_params:
                    # 连续参数变异
                    min_val, max_val = self.continuous_params[param_name]
                    
                    if param_name in self.log_scale_params:
                        # 对数空间变异
                        log_min, log_max = np.log10(min_val), np.log10(max_val)
                        current_log = np.log10(value)
                        # 高斯变异
                        mutation_strength = (log_max - log_min) * 0.1
                        new_log = current_log + np.random.normal(0, mutation_strength)
                        new_log = np.clip(new_log, log_min, log_max)
                        mutated_params[param_name] = 10 ** new_log
                    else:
                        # 线性空间变异
                        mutation_strength = (max_val - min_val) * 0.1
                        new_value = value + np.random.normal(0, mutation_strength)
                        mutated_params[param_name] = np.clip(new_value, min_val, max_val)

        return Individual(params=mutated_params, generation=individual.generation + 1)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作 - 支持离散和连续参数"""
        child1_params = copy.deepcopy(parent1.params)
        child2_params = copy.deepcopy(parent2.params)

        if random.random() < self.crossover_rate:
            for param_name in self.search_space.keys():
                if random.random() < 0.5:
                    # 简单交换
                    child1_params[param_name], child2_params[param_name] = \
                        child2_params[param_name], child1_params[param_name]
        
        # 确保整数列表参数是整数类型
        for param_name in self.integer_list_params.keys():
            if param_name in child1_params:
                child1_params[param_name] = int(child1_params[param_name])
            if param_name in child2_params:
                child2_params[param_name] = int(child2_params[param_name])

        child1 = Individual(params=child1_params, generation=max(parent1.generation, parent2.generation) + 1)
        child2 = Individual(params=child2_params, generation=max(parent1.generation, parent2.generation) + 1)

        return child1, child2

    def _select_parents(self) -> Tuple[Individual, Individual]:
        """锦标赛选择"""
        tournament_size = 3

        def tournament_selection():
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            return max(tournament, key=lambda x: x.fitness)

        parent1 = tournament_selection()
        parent2 = tournament_selection()
        return parent1, parent2

    def optimize(self, fitness_function: Callable[[Dict[str, Any]], float],
                 save_path: str = None) -> Dict[str, Any]:
        """执行遗传算法优化 - 增强错误处理"""
        print("=" * 60)
        print("🧬 Starting Genetic Algorithm Hyperparameter Optimization")
        print("=" * 60)

        # 初始化种群
        self._initialize_population()

        # 评估初始种群
        print(f"📊 Evaluating initial population ({self.population_size} individuals)...")
        valid_individuals = 0
        for i, individual in enumerate(self.population):
            print(f"   Individual {i + 1}/{self.population_size}: ", end="", flush=True)
            self._print_params_compact(individual.params)
            
            try:
                individual.fitness = fitness_function(individual.params)
                if not (np.isnan(individual.fitness) or np.isinf(individual.fitness)):
                    valid_individuals += 1
                print(f" → Fitness: {individual.fitness:.4f}")
            except Exception as e:
                print(f" → Error: {str(e)[:50]}...")
                individual.fitness = float('-inf')

        # 如果所有个体都失败，生成新的种群
        if valid_individuals == 0:
            print("\n⚠️  All individuals failed in initial population. Generating new population with safer parameters...")
            self._generate_safer_population()
            
            for i, individual in enumerate(self.population):
                print(f"   Retry Individual {i + 1}/{self.population_size}: ", end="", flush=True)
                self._print_params_compact(individual.params)
                
                try:
                    individual.fitness = fitness_function(individual.params)
                    if not (np.isnan(individual.fitness) or np.isinf(individual.fitness)):
                        valid_individuals += 1
                    print(f" → Fitness: {individual.fitness:.4f}")
                except Exception as e:
                    print(f" → Error: {str(e)[:50]}...")
                    individual.fitness = float('-inf')

        self.population_history_for_csv.append(copy.deepcopy(self.population))

        # 记录最佳个体
        valid_population = [ind for ind in self.population if not (np.isnan(ind.fitness) or np.isinf(ind.fitness))]
        if valid_population:
            self.best_individual = max(valid_population, key=lambda x: x.fitness)
        else:
            # 如果仍然没有有效个体，使用最保守的参数
            print("\n❌ No valid individuals found. Using most conservative parameters...")
            self.best_individual = Individual(params=self._get_most_conservative_params())
            self.best_individual.fitness = float('-inf')
            
        self.fitness_history.append(self.best_individual.fitness)

        print(f"\n🏆 Generation 0 Complete")
        print(f"   Valid individuals: {valid_individuals}/{self.population_size}")
        print(f"   Best Fitness: {self.best_individual.fitness:.4f}")
        self._print_best_params(self.best_individual.params)
        print("-" * 60)

        # 如果初始种群太少有效个体，减少进化代数
        if valid_individuals < self.population_size // 2:
            print(f"⚠️  Low success rate. Reducing generations from {self.generations} to {max(1, self.generations // 2)}")
            self.generations = max(1, self.generations // 2)

        # 进化过程
        for generation in range(1, self.generations + 1):
            print(f"\n🔄 Generation {generation}/{self.generations}")

            new_population = []

            # 保留精英（只保留有效的）
            sorted_population = sorted([ind for ind in self.population if not (np.isnan(ind.fitness) or np.isinf(ind.fitness))], 
                                     key=lambda x: x.fitness, reverse=True)
            elite_count = min(self.elite_ratio, len(sorted_population))
            elite = sorted_population[:elite_count]
            new_population.extend(elite)
            print(f"   ✅ Keeping {len(elite)} elite individuals")

            # 生成新个体
            offspring_count = 0
            max_attempts = self.population_size * 3  # 限制尝试次数
            attempt_count = 0
            
            while len(new_population) < self.population_size and attempt_count < max_attempts:
                attempt_count += 1
                
                # 选择父代
                if len(sorted_population) >= 2:
                    parent1, parent2 = self._select_parents_from_valid(sorted_population)
                else:
                    # 如果有效个体太少，生成新的随机个体
                    parent1 = Individual(params=self._generate_conservative_params())
                    parent2 = Individual(params=self._generate_conservative_params())

                child1, child2 = self._crossover(parent1, parent2)

                # 变异
                child1 = self._mutate(child1)
                if len(new_population) + 1 < self.population_size:
                    child2 = self._mutate(child2)

                # 评估适应度
                for child in [child1, child2]:
                    if len(new_population) >= self.population_size:
                        break
                        
                    offspring_count += 1
                    print(f"   Offspring {offspring_count}: ", end="", flush=True)
                    self._print_params_compact(child.params)
                    
                    try:
                        child.fitness = fitness_function(child.params)
                        print(f" → {child.fitness:.4f}")
                    except Exception as e:
                        print(f" → Error: {str(e)[:30]}...")
                        child.fitness = float('-inf')
                        
                    new_population.append(child)

            self.population = new_population[:self.population_size]
            self.population_history_for_csv.append(copy.deepcopy(self.population))

            # 更新最佳个体 - 修复improvement变量未定义的问题
            valid_current = [ind for ind in self.population if not (np.isnan(ind.fitness) or np.isinf(ind.fitness))]
            improvement = ""  # 初始化improvement变量在正确位置
            
            if valid_current:
                current_best = max(valid_current, key=lambda x: x.fitness)
                if current_best.fitness > self.best_individual.fitness:
                    improvement = f" (↑{current_best.fitness - self.best_individual.fitness:.4f})"
                    self.best_individual = current_best

            self.fitness_history.append(self.best_individual.fitness)

            print(f"\n🏆 Generation {generation} Complete")
            print(f"   Valid individuals: {len(valid_current)}/{self.population_size}")
            print(f"   Best Fitness: {self.best_individual.fitness:.4f}{improvement}")
            self._print_best_params(self.best_individual.params)
            
            # 保存中间结果 - 每代都保存
            if save_path:
                self._save_results(save_path, generation)
                print(f"   💾 Results saved to {save_path}")

            # 调用回调函数 - 每代都调用，用于保存图表
            if self.on_generation_end:
                self.on_generation_end(generation, self.population, self.best_individual)

            print("-" * 60)

        # 保存最终结果
        if save_path:
            self._save_results(save_path, self.generations)

        print("\n" + "=" * 60)
        print("🎯 OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print("🏆 BEST HYPERPARAMETERS FOUND:")
        self._print_best_params(self.best_individual.params, detailed=True)
        print(f"🎖️  BEST FITNESS: {self.best_individual.fitness:.6f}")
        print("=" * 60)

        return self.best_individual.params

    def _save_results(self, save_path: str, generation: int):
        """保存优化结果"""
        os.makedirs(save_path, exist_ok=True)

        results = {
            'best_params': self.best_individual.params,
            'best_fitness': self.best_individual.fitness,
            'generation': generation,
            'fitness_history': self.fitness_history,
            'search_space': self.search_space,
            'optimization_config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_ratio': self.elite_ratio,
                'random_seed': self.random_seed
            }
        }

        with open(os.path.join(save_path, f'optimization_results_gen_{generation}.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _print_params_compact(self, params: Dict[str, Any]):
        """紧凑打印参数 - 改进离散参数显示"""
        param_str = []
        for k, v in params.items():
            if isinstance(v, float):
                if k in self.log_scale_params:
                    param_str.append(f"{k}={v:.2e}")
                else:
                    param_str.append(f"{k}={v:.4g}")
            else:
                param_str.append(f"{k}={v}")
        print(f"[{', '.join(param_str)}]", end="")

    def _print_best_params(self, params: Dict[str, Any], detailed: bool = False):
        """打印最佳参数 - 改进格式"""
        if detailed:
            print("\n" + "="*50)
            for key, value in params.items():
                if isinstance(value, float):
                    if key in self.log_scale_params:
                        print(f"   📋 {key}: {value:.6e}")
                    else:
                        print(f"   📋 {key}: {value:.6f}")
                else:
                    print(f"   📋 {key}: {value}")
            print("="*50)
        else:
            param_strings = []
            for key, value in params.items():
                if isinstance(value, float):
                    if key in self.log_scale_params:
                        param_strings.append(f"{key}={value:.3e}")
                    else:
                        param_strings.append(f"{key}={value:.4f}")
                else:
                    param_strings.append(f"{key}={value}")
            print(f"   📋 {' | '.join(param_strings)}")

    def _generate_safer_population(self):
        """生成更安全的参数种群"""
        self.population = []
        for _ in range(self.population_size):
            params = self._generate_conservative_params()
            individual = Individual(params=params, generation=0)
            self.population.append(individual)

    def _generate_conservative_params(self) -> Dict[str, Any]:
        """生成保守参数 - ✅ 新增 mlp_ratio 等关键参数支持"""
        params = {}
        
        # === 关键新增：显式定义所有参数的保守值 ===
        CONSERVATIVE_DEFAULTS = {
            # 模型架构参数
            'mlp_ratio': 4.0,
            'num_pool': 4,
            'transformer_layers': 3,
            'transformer_hidden_size': 128,
            'transformer_heads': 4,
            'transformer_layers_num': 1,       # ContextUnet 架构搜索
            'num_global_tokens': 4,            # ContextUnet 架构搜索
            'arch_dropout': 0.1,               # ContextUnet 架构搜索
            'arch_drop_path': 0.1,             # ContextUnet 架构搜索

            # 优化器参数
            'optimizer_type': 'AdamW',
            'initial_lr': 5e-5,
            'weight_decay': 1e-6,
            'eps': 1e-8,
            'beta1': 0.9,
            'beta2': 0.999,
            'dropout_rate': 0.0,
            'oversample_foreground_percent': 0.33,
        }
        
        # 遍历搜索空间中所有参数
        for param_name in self.search_space.keys():
            if param_name in CONSERVATIVE_DEFAULTS:
                params[param_name] = CONSERVATIVE_DEFAULTS[param_name]
            else:
                # 通用回退逻辑（保持原有）
                if param_name in self.discrete_params:
                    params[param_name] = self.discrete_params[param_name][0]
                elif param_name in self.integer_list_params:
                    params[param_name] = int(self.integer_list_params[param_name][0])
                elif param_name in self.continuous_params:
                    min_val, max_val = self.continuous_params[param_name]
                    if param_name in self.log_scale_params:
                        params[param_name] = 10 ** np.mean([np.log10(min_val), np.log10(max_val)])
                    else:
                        params[param_name] = (min_val + max_val) / 2
                else:
                    params[param_name] = 0  # 安全兜底
        
        # === 关键：强制类型校验（防整数变浮点）===
        for int_param in ['num_pool', 'transformer_layers', 'transformer_heads',
                          'transformer_layers_num', 'transformer_hidden_size',
                          'num_global_tokens']:
            if int_param in params:
                params[int_param] = int(params[int_param])
        
        return params

    def _get_most_conservative_params(self) -> Dict[str, Any]:
        return {
            'optimizer_type': 'AdamW',
            'initial_lr': 5e-5,
            'weight_decay': 1e-7,
            'oversample_foreground_percent': 0.33,
            'eps': 1e-8,
            'beta1': 0.9,
            'beta2': 0.999,
            'amsgrad': False,
            'dropout_rate': 0.0,
            # ContextUnet 架构保守值
            'transformer_hidden_size': 128,
            'transformer_heads': 4,
            'transformer_layers_num': 1,
            'num_global_tokens': 4,
            'arch_dropout': 0.1,
            'arch_drop_path': 0.1,
        }

    def _select_parents_from_valid(self, valid_population):
        """从有效种群中选择父代"""
        if len(valid_population) < 2:
            return valid_population[0], valid_population[0]
            
        tournament_size = min(3, len(valid_population))
        
        def tournament_selection():
            tournament = random.sample(valid_population, tournament_size)
            return max(tournament, key=lambda x: x.fitness)
        
        parent1 = tournament_selection()
        parent2 = tournament_selection()
        return parent1, parent2


