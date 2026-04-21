#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入现有系统的模块
from V9EAPSO import Config, TiffTerrainLoader, DronePSOOptimizer

# ======================== 日志配置 ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ======================== 算法基类 ========================
class BaseOptimizer:
    """所有优化算法的基类，提供通用接口和评估方法"""
    
    def __init__(self, terrain, building_candidates, task_points):
        self.terrain = terrain
        self.building_candidates = building_candidates  # shape (M,3)
        self.task_points = task_points
        self.bc_kdtree = KDTree(building_candidates[:, :2])
        self.num_stations = Config.NUM_STATIONS
        self.best_solution = None
        self.best_score = -np.inf
        self.execution_time = 0
        self.los_penalty_params = None # 初始化LoS惩罚参数

    def project_to_candidate(self, p):
        """将点投影到最近的建筑物候选点"""
        dist, ind = self.bc_kdtree.query(np.array([p]), return_distance=True)
        return self.building_candidates[ind[0][0], :2]
    
    def get_docking_altitude(self, p):
        """获取对接点的高度"""
        dist, ind = self.bc_kdtree.query(np.array([p]), return_distance=True)
        return self.building_candidates[ind[0][0], 2]
    
    def has_line_of_sight(self, p1, p2):
        """检查两点之间是否有视线"""
        a1 = self.get_docking_altitude(p1)
        a2 = self.get_docking_altitude(p2)
        p1_3d = np.array([p1[0], p1[1], a1])
        p2_3d = np.array([p2[0], p2[1], a2])
        dx = p2_3d[0] - p1_3d[0]
        dy = p2_3d[1] - p1_3d[1]
        dz = p2_3d[2] - p1_3d[2]
        distance = np.sqrt(dx**2 + dy**2)
        step_size = 5.0
        steps = max(int(distance/step_size), 1)
        for i in range(steps+1):
            x = p1_3d[0] + dx * i / steps
            y = p1_3d[1] + dy * i / steps
            z_line = p1_3d[2] + dz * i / steps
            terrain_z = self.terrain.get_elevation(x, y)
            if z_line < terrain_z + 10:
                return False
        return True

    @staticmethod
    def _compute_los_penalty_piecewise(broken: int, num_stations: int) -> float:
        """分段化视线（LoS）惩罚函数

        参数:
            broken: 基站对中不可见（断线）的数量。
            num_stations: 基站数量，用于计算成对组合的总数。

        返回:
            惩罚值，范围在 [0, 100] 之间，随不可见比例单调递增并在高遮挡时饱和。

        说明:
            将不可见对的占比分段线性映射为惩罚值，避免阶跃式大惩罚导致总分失真。
            分段阈值可根据场景调整，这里选择四段：<=5%、<=20%、<=50%、>50%。
        """
        total_pairs = max(num_stations * (num_stations - 1) // 2, 1)
        frac = broken / total_pairs

        if frac <= 0.05:
            penalty = 200.0 * frac  # 0 → 10
        elif frac <= 0.20:
            penalty = 10.0 + (frac - 0.05) * (30.0 / 0.15)
        elif frac <= 0.50:
            penalty = 40.0 + (frac - 0.20) * (40.0 / 0.30)
        else:
            penalty = 80.0 + (frac - 0.50) * (20.0 / 0.50)

        return max(0.0, min(100.0, float(penalty)))
    
    def compute_fitness(self, solution):
        """计算适应度函数（覆盖率 - 能量惩罚 - 分段化LoS惩罚）

        说明:
            - 覆盖率：任务点在覆盖半径内的比例，映射到百分制。
            - 能量惩罚：超过电池容量的能量消耗按比例加权。
            - LoS惩罚：对基站对的不可见比例进行分段线性映射，范围限定在[0,100]，避免惩罚爆炸。
        """
        # 确保solution是二维数组，形状为(num_stations, 2)
        solution = np.array(solution).reshape(-1, 2)
        
        # 计算每个任务点到最近基站的距离
        diffs = self.task_points[:, np.newaxis, :] - solution[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        
        # 计算覆盖率
        covered = min_dists <= Config.UAV_COVERAGE_RADIUS
        coverage_ratio = np.sum(covered) / len(self.task_points) * 100.0
        
        # 计算能量消耗
        flight_times = np.sqrt(min_dists[covered]**2 + Config.UAV_DEPLOYMENT_ALTITUDE**2) / Config.UAV_SPEED
        energies = Config.P_m * flight_times
        energy_excess = np.maximum(energies - Config.BATTERY_CAPACITY, 0)
        energy_penalty = np.sum(energy_excess) * Config.ENERGY_PENALTY_FACTOR
        
        # 计算视线惩罚（分段化、平滑且可控）
        n = solution.shape[0]
        broken = 0
        for i in range(n):
            for j in range(i+1, n):
                if not self.has_line_of_sight(solution[i], solution[j]):
                    broken += 1
        los_penalty = self._compute_los_penalty_piecewise(broken, n)
        
        # 总适应度
        fitness = coverage_ratio - energy_penalty - los_penalty
        return fitness
    
    def get_coverage_efficiency(self, solution):
        """计算覆盖效率（百分比）"""
        solution = np.array(solution).reshape(-1, 2)
        diffs = self.task_points[:, np.newaxis, :] - solution[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        covered = min_dists <= Config.UAV_COVERAGE_RADIUS
        return np.sum(covered) / len(self.task_points) * 100.0
    
    def optimize(self):
        """优化方法，由子类实现"""
        raise NotImplementedError("子类必须实现optimize方法")

# ======================== K-means聚类算法 ========================
class KMeansOptimizer(BaseOptimizer):
    """基于K-means聚类的优化器"""
    
    def optimize(self):
        start_time = time.time()
        
        # 使用K-means聚类任务点
        kmeans = KMeans(n_clusters=self.num_stations, random_state=42, n_init=10)
        kmeans.fit(self.task_points)
        
        # 获取聚类中心
        centers = kmeans.cluster_centers_
        
        # 将聚类中心投影到最近的建筑物候选点
        solution = np.zeros((self.num_stations, 2))
        for i in range(self.num_stations):
            solution[i] = self.project_to_candidate(centers[i])
        
        # 计算适应度
        fitness = self.compute_fitness(solution)
        
        self.best_solution = solution
        self.best_score = fitness
        self.execution_time = time.time() - start_time
        
        logger.info(f"K-means优化完成，适应度: {self.best_score:.2f}，耗时: {self.execution_time:.2f}秒")
        return self.best_solution, self.best_score

# ======================== 遗传算法 ========================
class GeneticOptimizer(BaseOptimizer):
    """基于遗传算法的优化器"""
    
    def optimize(self):
        start_time = time.time()
        
        # 创建适应度和个体类型
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # 初始化工具箱
        toolbox = base.Toolbox()
        
        # 定义个体和种群
        def create_individual():
            # 随机选择建筑物候选点
            indices = np.random.choice(len(self.building_candidates), self.num_stations, replace=False)
            return creator.Individual([self.building_candidates[i, :2].tolist() for i in indices])
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # 定义评估函数
        def evaluate(individual):
            solution = np.array([point for point in individual])
            return (self.compute_fitness(solution),)
        
        toolbox.register("evaluate", evaluate)
        
        # 定义遗传操作
        def custom_crossover(ind1, ind2):
            # 随机交换部分基站位置
            for i in range(self.num_stations):
                if np.random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2
        
        def custom_mutate(individual):
            # 随机替换一些基站位置
            for i in range(self.num_stations):
                if np.random.random() < 0.2:  # 20%的变异概率
                    idx = np.random.randint(0, len(self.building_candidates))
                    individual[i] = self.building_candidates[idx, :2].tolist()
            return individual,
        
        toolbox.register("mate", custom_crossover)
        toolbox.register("mutate", custom_mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 创建初始种群
        pop_size = Config.PSO_PARTICLES
        population = toolbox.population(n=pop_size)
        
        # 记录统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 运行遗传算法
        pop, logbook = algorithms.eaSimple(
            population, toolbox, 
            cxpb=0.7,  # 交叉概率
            mutpb=0.2,  # 变异概率
            ngen=Config.MAX_ITERATIONS,  # 迭代次数
            stats=stats,
            verbose=True
        )
        
        # 获取最佳个体
        best_ind = tools.selBest(pop, 1)[0]
        self.best_solution = np.array([point for point in best_ind])
        self.best_score = self.compute_fitness(self.best_solution)
        self.execution_time = time.time() - start_time
        
        logger.info(f"遗传算法优化完成，适应度: {self.best_score:.2f}，耗时: {self.execution_time:.2f}秒")
        return self.best_solution, self.best_score

# ======================== NSGA-II多目标优化算法 ========================
class NSGAIIOptimizer(BaseOptimizer):
    """基于NSGA-II的多目标优化器"""
    
    def optimize(self):
        start_time = time.time()
        
        # 定义多目标优化问题
        class DroneDeploymentProblem(Problem):
            def __init__(self, optimizer):
                self.optimizer = optimizer
                
                # 定义变量边界
                x_min = np.min(optimizer.building_candidates[:, 0])
                x_max = np.max(optimizer.building_candidates[:, 0])
                y_min = np.min(optimizer.building_candidates[:, 1])
                y_max = np.max(optimizer.building_candidates[:, 1])
                
                xl = np.array([x_min, y_min] * optimizer.num_stations)
                xu = np.array([x_max, y_max] * optimizer.num_stations)
                
                # 只调用一次父类初始化方法
                super().__init__(n_var=optimizer.num_stations * 2, 
                                 n_obj=3, 
                                 n_constr=0, 
                                 xl=xl, 
                                 xu=xu)
            
            def _evaluate(self, x, out, *args, **kwargs):
                # 计算多个目标
                n_points = x.shape[0]  # 种群大小
                f1 = np.zeros(n_points)  # 覆盖率（最大化）
                f2 = np.zeros(n_points)  # 能量消耗（最小化）
                f3 = np.zeros(n_points)  # 视线阻断（最小化）
                
                for i in range(n_points):
                    # 重塑解决方案
                    solution = x[i].reshape(self.optimizer.num_stations, 2)
                    
                    # 将解投影到建筑物候选点
                    for j in range(self.optimizer.num_stations):
                        solution[j] = self.optimizer.project_to_candidate(solution[j])
                    
                    # 计算覆盖率
                    diffs = self.optimizer.task_points[:, np.newaxis, :] - solution[np.newaxis, :, :]
                    dists = np.linalg.norm(diffs, axis=2)
                    min_dists = dists.min(axis=1)
                    covered = min_dists <= Config.UAV_COVERAGE_RADIUS
                    coverage_ratio = np.sum(covered) / len(self.optimizer.task_points) * 100.0
                    
                    # 计算能量消耗
                    flight_times = np.sqrt(min_dists[covered]**2 + Config.UAV_DEPLOYMENT_ALTITUDE**2) / Config.UAV_SPEED
                    energies = Config.P_m * flight_times
                    energy_excess = np.maximum(energies - Config.BATTERY_CAPACITY, 0)
                    energy_penalty = np.sum(energy_excess) * Config.ENERGY_PENALTY_FACTOR
                    
                    # 计算视线阻断
                    los_broken = 0
                    for j in range(self.optimizer.num_stations):
                        for k in range(j+1, self.optimizer.num_stations):
                            if not self.optimizer.has_line_of_sight(solution[j], solution[k]):
                                los_broken += 1
                    
                    # 设置目标值（注意：pymoo默认是最小化所有目标）
                    f1[i] = -coverage_ratio  # 取负值以便最小化
                    f2[i] = energy_penalty
                    f3[i] = los_broken
                
                out["F"] = np.column_stack([f1, f2, f3])
        
        # 创建问题实例
        problem = DroneDeploymentProblem(self)
        
        # 配置算法
        algorithm = NSGA2(
            pop_size=Config.PSO_PARTICLES,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # 运行优化
        res = minimize(
            problem,
            algorithm,
            ('n_gen', Config.MAX_ITERATIONS),
            verbose=True
        )
        
        # 从帕累托前沿选择一个解
        # 这里我们选择覆盖率最高的解
        best_idx = np.argmin(res.F[:, 0])  # 第一列是覆盖率的负值
        best_x = res.X[best_idx]
        
        # 重塑解并投影到建筑物候选点
        solution = best_x.reshape(self.num_stations, 2)
        for i in range(self.num_stations):
            solution[i] = self.project_to_candidate(solution[i])
        
        self.best_solution = solution
        self.best_score = self.compute_fitness(solution)
        self.execution_time = time.time() - start_time
        
        logger.info(f"NSGA-II优化完成，适应度: {self.best_score:.2f}，耗时: {self.execution_time:.2f}秒")
        return self.best_solution, self.best_score

# ======================== 蚁群算法 ========================
class AntColonyOptimizer(BaseOptimizer):
    """基于蚁群算法的优化器"""
    
    def optimize(self):
        start_time = time.time()
        
        # 蚁群算法参数
        n_ants = Config.PSO_PARTICLES
        n_iterations = Config.MAX_ITERATIONS
        alpha = 1.0  # 信息素重要程度
        beta = 2.0   # 启发式因子重要程度
        rho = 0.5    # 信息素蒸发率
        Q = 100      # 信息素增加强度
        
        # 创建候选点网格
        n_candidates = len(self.building_candidates)
        
        # 初始化信息素矩阵
        pheromone = np.ones((n_candidates, n_candidates))
        
        # 计算启发式信息（基于距离的倒数）
        heuristic = np.zeros((n_candidates, n_candidates))
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    dist = np.linalg.norm(self.building_candidates[i, :2] - self.building_candidates[j, :2])
                    heuristic[i, j] = 1.0 / (dist + 1e-10)
        
        # 初始化最佳解
        best_solution_indices = None
        best_fitness = -np.inf
        
        # 迭代优化
        for iteration in tqdm(range(n_iterations), desc="蚁群算法迭代"):
            # 每只蚂蚁构建解
            all_solutions = []
            all_fitnesses = []
            
            for ant in range(n_ants):
                # 随机选择起始点
                current = np.random.randint(0, n_candidates)
                solution_indices = [current]
                
                # 构建完整解
                for _ in range(self.num_stations - 1):
                    # 计算转移概率
                    probabilities = np.zeros(n_candidates)
                    for j in range(n_candidates):
                        if j not in solution_indices:
                            probabilities[j] = (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)
                    
                    # 归一化概率
                    if np.sum(probabilities) > 0:
                        probabilities = probabilities / np.sum(probabilities)
                    
                    # 轮盘赌选择下一个点
                    next_point = np.random.choice(n_candidates, p=probabilities)
                    solution_indices.append(next_point)
                    current = next_point
                
                # 构建解决方案
                solution = self.building_candidates[solution_indices, :2]
                
                # 计算适应度
                fitness = self.compute_fitness(solution)
                
                all_solutions.append(solution_indices)
                all_fitnesses.append(fitness)
                
                # 更新最佳解
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution_indices = solution_indices.copy()
            
            # 更新信息素
            pheromone = pheromone * (1 - rho)  # 信息素蒸发
            
            # 每只蚂蚁留下信息素
            for ant, (solution_indices, fitness) in enumerate(zip(all_solutions, all_fitnesses)):
                if fitness > 0:  # 只有正适应度的解才增加信息素
                    for i in range(len(solution_indices) - 1):
                        from_idx = solution_indices[i]
                        to_idx = solution_indices[i + 1]
                        pheromone[from_idx, to_idx] += Q * fitness
                        pheromone[to_idx, from_idx] += Q * fitness  # 对称更新
        
        # 构建最终解决方案
        self.best_solution = self.building_candidates[best_solution_indices, :2]
        self.best_score = best_fitness
        self.execution_time = time.time() - start_time
        
        logger.info(f"蚁群算法优化完成，适应度: {self.best_score:.2f}，耗时: {self.execution_time:.2f}秒")
        return self.best_solution, self.best_score

# ======================== PSO算法包装类 ========================
class PSOOptimizerWrapper(BaseOptimizer):
    """对现有PSO优化器的包装类"""
    
    def optimize(self):
        start_time = time.time()
        
        # 使用现有的PSO优化器
        pso = DronePSOOptimizer(self.terrain, self.building_candidates, self.task_points)
        self.best_solution, self.best_score = pso.optimize()
        self.execution_time = time.time() - start_time
        
        logger.info(f"PSO优化完成，适应度: {self.best_score:.2f}，耗时: {self.execution_time:.2f}秒")
        return self.best_solution, self.best_score

# ======================== 算法比较模块 ========================
class AlgorithmComparison:
    """算法比较类，用于运行和比较不同算法的性能"""
    
    def __init__(self, terrain, building_candidates, task_points, buildings=None, roads=None):
        self.terrain = terrain
        self.building_candidates = building_candidates
        self.task_points = task_points
        self.buildings = buildings  # 保存建筑物数据用于可视化
        self.roads = roads  # 保存路网数据用于可视化
        self.algorithms = {
            'K-means': KMeansOptimizer,
            'GA': GeneticOptimizer,
            'NSGA-II': NSGAIIOptimizer,
            'ACO': AntColonyOptimizer,
            'PSO': PSOOptimizerWrapper
        }
        self.results = {}
        self.bc_kdtree = KDTree(building_candidates[:, :2]) if building_candidates is not None else None
    
    def run_comparison(self, algorithms=None):
        """运行指定算法的比较"""
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        for algo_name in algorithms:
            if algo_name not in self.algorithms:
                logger.warning(f"未知算法: {algo_name}，跳过")
                continue
            
            logger.info(f"运行{algo_name}算法...")
            optimizer = self.algorithms[algo_name](self.terrain, self.building_candidates, self.task_points)
            solution, fitness = optimizer.optimize()
            coverage = optimizer.get_coverage_efficiency(solution)
            
            # 记录算法结果
            self.results[algo_name] = {
                'solution': solution.tolist(),  # 转换为列表以便JSON序列化
                'fitness': fitness,
                'coverage': coverage,
                'time': optimizer.execution_time
            }
            
            logger.info(f"{algo_name}算法结果: 适应度={fitness:.2f}, 覆盖率={coverage:.2f}%, 耗时={optimizer.execution_time:.2f}秒")
        
        return self.results
        
    def save_results(self, output_file="../data_results/algorithm_results.json"):
        """将算法结果保存到JSON文件
        
        Args:
            output_file: 输出JSON文件路径
        """
        # 准备元数据
        metadata = {
            'config': {
                'UAV_COVERAGE_RADIUS': Config.UAV_COVERAGE_RADIUS,
                'UAV_DEPLOYMENT_ALTITUDE': Config.UAV_DEPLOYMENT_ALTITUDE,
                'NUM_STATIONS': Config.NUM_STATIONS,
                'NUM_TASK_POINTS': Config.NUM_TASK_POINTS
            },
            'terrain_path': self.terrain.tiff_path if hasattr(self.terrain, 'tiff_path') else None,
            'building_candidates': self.building_candidates.tolist() if self.building_candidates is not None else None,
            'task_points': self.task_points.tolist() if self.task_points is not None else None
        }
        
        # 准备完整数据
        data = {
            'metadata': metadata,
            'results': self.results
        }
        
        # 保存到文件
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"算法结果已保存至: {output_file}")
            return True
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            return False
        
        return self.results
    
    def plot_comparison(self, save_path=None):
        """绘制算法比较结果，包含数据标准化处理和丰富的图表类型"""
        if not self.results:
            logger.warning("没有可用的比较结果，请先运行run_comparison()")
            return
        
        # 设置中文字体支持
        try:
            import matplotlib.font_manager as fm
            # 尝试使用系统中的中文字体
            font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            chinese_fonts = [f for f in font_paths if 'simhei' in f.lower() or 'msyh' in f.lower() or 'simsun' in f.lower()]
            if chinese_fonts:
                plt.rcParams['font.family'] = fm.FontProperties(fname=chinese_fonts[0]).get_name()
            else:
                # 如果没有找到中文字体，使用默认字体并禁用中文警告
                plt.rcParams['font.family'] = 'sans-serif'
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
        except Exception as e:
            logger.warning(f"设置中文字体失败: {str(e)}，将使用默认字体")
            plt.rcParams['font.family'] = 'sans-serif'
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
        
        # 提取结果数据
        algo_names = list(self.results.keys())
        fitness_values = [self.results[name]['fitness'] for name in algo_names]
        coverage_values = [self.results[name]['coverage'] for name in algo_names]
        time_values = [self.results[name]['time'] for name in algo_names]
        
        # 数据标准化处理
        # 1. 计算各指标的最大值和最小值
        max_fitness = max(fitness_values) if fitness_values else 1.0
        min_fitness = min(fitness_values) if fitness_values else 0.0
        max_coverage = 100  # 覆盖率最大为100%
        min_coverage = 0    # 覆盖率最小为0%
        max_time = max(time_values) if time_values else 1.0
        min_time = min(time_values) if time_values else 0.1
        
        # 2. 标准化数据到[0,1]区间
        norm_fitnesses = [(f - min_fitness) / (max_fitness - min_fitness) if max_fitness != min_fitness else 0.5 for f in fitness_values]
        norm_coverages = [c / 100 for c in coverage_values]  # 覆盖率已经是百分比
        # 时间取倒数标准化，因为时间越短越好
        norm_time_efficiency = [1 - ((t - min_time) / (max_time - min_time)) if max_time != min_time else 0.5 for t in time_values]
        
        # 创建图表 - 使用2x2布局
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 适应度柱状图 (左上)
        ax1 = fig.add_subplot(2, 2, 1)
        # 使用标准化后的颜色渐变
        colors1 = plt.cm.Blues(np.array(norm_fitnesses))
        bars1 = ax1.bar(algo_names, [r['normalized_fitness_relative'] for r in self.results.values()], color=colors1, width=0.6)
        ax1.set_title('fitness comparison', fontsize=16)
        ax1.set_ylabel('Normalized Fitness (Relative)', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.tick_params(axis='both', labelsize=12)
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_fitness,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        # 2. 覆盖率柱状图 (右上)
        ax2 = fig.add_subplot(2, 2, 2)
        # 使用标准化后的颜色渐变
        colors2 = plt.cm.Greens(np.array(norm_coverages))
        bars2 = ax2.bar(algo_names, coverage_values, color=colors2, width=0.6)
        ax2.set_title('coverage comparison (%)', fontsize=16)
        ax2.set_ylabel('coverage  (%)', fontsize=14)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', labelsize=12)
        # 在柱状图上添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
        
        # 3. 热力图 - 综合性能比较 (左下)
        ax3 = fig.add_subplot(2, 2, 3)
        # 准备热力图数据
        heatmap_data = np.array([norm_fitnesses, norm_coverages, norm_time_efficiency])
        im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
        
        # 设置热力图标签
        ax3.set_yticks(np.arange(3))
        ax3.set_yticklabels(['fitness', 'coverage', 'efficiency'], fontsize=12)
        ax3.set_xticks(np.arange(len(algo_names)))
        ax3.set_xticklabels(algo_names, fontsize=12)
        
        # 添加热力图数值标签
        for i in range(3):
            for j in range(len(algo_names)):
                if i == 0:
                    text = f'{fitness_values[j]:.2f}'
                elif i == 1:
                    text = f'{coverage_values[j]:.2f}%'
                else:
                    text = f'{time_values[j]:.2f}s'
                ax3.text(j, i, text, ha="center", va="center", color="white" if heatmap_data[i, j] < 0.7 else "black", fontsize=10)
        
        ax3.set_title('Comprehensive Algorithm Performance Heatmap', fontsize=16)
        fig.colorbar(im, ax=ax3, label='Normalized Performance Score')
        
        # 4. 雷达图 - 综合比较 (右下)
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
        
        # 设置雷达图的角度
        categories = ['fitness', 'coverage', 'efficiency']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图
        
        # 绘制雷达图
        ax4.set_theta_offset(np.pi / 2)  # 从顶部开始
        ax4.set_theta_direction(-1)  # 顺时针
        
        # 设置雷达图的刻度标签
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=14)
        
        # 绘制每个算法的雷达图
        for i, algo in enumerate(algo_names):
            values = [norm_fitnesses[i], norm_coverages[i], norm_time_efficiency[i]]
            values += values[:1]  # 闭合雷达图
            ax4.plot(angles, values, linewidth=2, label=algo)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_title('Comprehensive Algorithm Performance Radar Chart', fontsize=16)
        # 将图例放在图表下方，避免遮挡
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)
        
        # 添加标题和调整布局
        plt.suptitle('Multi-Dimensional Comparative Analysis of Algorithm Performance', fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"增强版比较结果图表已保存至: {save_path}")
        
        return fig
    
    def visualize_solutions(self, html_filename="algorithm_comparison_3d.html"):
        """在3D环境中可视化不同算法的解决方案，使用2x2方形布局"""
        if not self.results:
            logger.warning("没有可用的比较结果，请先运行run_comparison()")
            return
        
        # 计算2x2布局的行列数
        num_algos = len(self.results)
        rows = 2
        cols = 2
        
        if num_algos > 4:
            logger.warning(f"算法数量({num_algos})超过4个，只显示前4个算法")
            algo_items = list(self.results.items())[:4]
        else:
            algo_items = list(self.results.items())
        
        # 创建子图 - 使用2x2布局
        subplot_titles = [algo_name for algo_name, _ in algo_items]
        specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=specs,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # 为每个算法创建一个场景
        for idx, (algo_name, result) in enumerate(algo_items):
            solution = np.array(result['solution'])
            coverage = result['coverage']
            
            # 地形表面：颜色从蓝到绿到黄到橙到红
            custom_colorscale = [
                [0.0, '#0000FF'], [0.25, '#00FF00'], [0.5, '#FFFF00'],
                [0.75, '#FFA500'], [1.0, '#FF0000']
            ]
            fig.add_trace(
                go.Surface(
                    x=self.terrain.utm_x,
                    y=self.terrain.utm_y,
                    z=self.terrain.elevation,
                    colorscale=custom_colorscale,
                    opacity=0.9,
                    showscale=(idx == 0),  # 只在第一个子图显示颜色条
                    colorbar=dict(title='Elevation (m)', x=-0.07) if idx == 0 else None
                ),
                # 计算在2x2网格中的位置
                row=idx // cols + 1,  # 行号从1开始
                col=idx % cols + 1    # 列号从1开始
            )
            
            # 建筑物：全部显示，颜色为灰色
            try:
                if hasattr(self, 'buildings') and not self.buildings.empty:
                    dem_poly = Polygon([
                        (float(np.min(self.terrain.utm_x)), float(np.min(self.terrain.utm_y))),
                        (float(np.max(self.terrain.utm_x)), float(np.min(self.terrain.utm_y))),
                        (float(np.max(self.terrain.utm_x)), float(np.max(self.terrain.utm_y))),
                        (float(np.min(self.terrain.utm_x)), float(np.max(self.terrain.utm_y)))
                    ])
                    valid_buildings = self.buildings[self.buildings.centroid.within(dem_poly)]
                    buildings_to_plot = valid_buildings.copy()
                    bottom_x, bottom_y, bottom_z = [], [], []
                    top_x, top_y, top_z = [], [], []
                    side_x, side_y, side_z = [], [], []
                    for b_idx, row in buildings_to_plot.iterrows():
                        geom = row.geometry
                        if geom is None:
                            continue
                        if geom.geom_type == "Polygon":
                            polys = [geom]
                        elif geom.geom_type == "MultiPolygon":
                            polys = list(geom.geoms)
                        else:
                            continue
                        height = 20.0
                        if 'Height' in buildings_to_plot.columns:
                            try:
                                h_val = float(row['Height'])
                                if h_val > 0:
                                    height = h_val
                            except Exception:
                                pass
                        rep = geom.representative_point()
                        base_z = self.terrain.get_elevation(rep.x, rep.y)
                        roof_z = base_z + height
                        for poly in polys:
                            exterior = poly.exterior.coords.xy
                            xs = list(exterior[0])
                            ys = list(exterior[1])
                            if len(xs) > 20:
                                xs = xs[::5]
                                ys = ys[::5]
                            bottom_x.extend(xs + [None])
                            bottom_y.extend(ys + [None])
                            bottom_z.extend([base_z]*len(xs) + [None])
                            top_x.extend(xs + [None])
                            top_y.extend(ys + [None])
                            top_z.extend([roof_z]*len(xs) + [None])
                            for x, y in zip(xs, ys):
                                side_x.extend([x, x, None])
                                side_y.extend([y, y, None])
                                side_z.extend([base_z, roof_z, None])
                    fig.add_trace(
                        go.Scatter3d(
                            x=bottom_x,
                            y=bottom_y,
                            z=bottom_z,
                            mode='lines',
                            line=dict(color='gray', width=2),
                            name='Building Bottom',
                            showlegend=(idx == 0),
                            opacity=0.7
                        ),
                        row=idx // cols + 1,
                        col=idx % cols + 1
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=top_x,
                            y=top_y,
                            z=top_z,
                            mode='lines',
                            line=dict(color='gray', width=2),
                            name='Building Top',
                            showlegend=(idx == 0),
                            opacity=0.7
                        ),
                        row=idx // cols + 1,
                        col=idx % cols + 1
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=side_x,
                            y=side_y,
                            z=side_z,
                            mode='lines',
                            line=dict(color='gray', width=1),
                            name='Building Side',
                            showlegend=(idx == 0),
                            opacity=0.7
                        ),
                        row=idx // cols + 1,
                        col=idx % cols + 1
                    )
            except Exception as e:
                logger.warning(f"绘制建筑物时出错: {str(e)}")
            
            # 路网：将所有路网数据合并为一条轨迹，灰色配色
            try:
                if hasattr(self, 'roads') and self.roads:
                    # 对每个路网坐标点，裁剪到 DEM 的范围内，避免浮空
                    min_x = np.min(self.terrain.utm_x)
                    max_x = np.max(self.terrain.utm_x)
                    min_y = np.min(self.terrain.utm_y)
                    max_y = np.max(self.terrain.utm_y)
                    all_road_x, all_road_y, all_road_z = [], [], []
                    for road in self.roads:
                        xs = list(road.xy[0])
                        ys = list(road.xy[1])
                        # 裁剪每个点到 DEM 范围内
                        xs = [np.clip(x, min_x, max_x) for x in xs]
                        ys = [np.clip(y, min_y, max_y) for y in ys]
                        zs = [self.terrain.get_elevation(x, y) + 2 for x, y in zip(xs, ys)]
                        all_road_x.extend(xs + [None])
                        all_road_y.extend(ys + [None])
                        all_road_z.extend(zs + [None])
                    fig.add_trace(
                        go.Scatter3d(
                            x=all_road_x,
                            y=all_road_y,
                            z=all_road_z,
                            mode='lines',
                            line=dict(color='gray', width=3),
                            name='Roads',
                            opacity=0.8,
                            showlegend=(idx == 0)
                        ),
                        row=idx // cols + 1,
                        col=idx % cols + 1
                    )
            except Exception as e:
                logger.warning(f"绘制路网时出错: {str(e)}")
            
            # 任务点：以绿色小点显示
            try:
                task_zs = [self.terrain.get_elevation(x, y) for x, y in self.task_points]
                fig.add_trace(
                    go.Scatter3d(
                        x=self.task_points[:,0].tolist(),
                        y=self.task_points[:,1].tolist(),
                        z=task_zs,
                        mode='markers',
                        marker=dict(size=3, color='green'),
                        name='Task Points',
                        opacity=0.8,
                        showlegend=(idx == 0)  # 只在第一个子图显示图例
                    ),
                    # 计算在2x2网格中的位置
                    row=idx // cols + 1,  # 行号从1开始
                    col=idx % cols + 1    # 列号从1开始
                )
            except Exception as e:
                logger.warning(f"绘制任务点时出错: {str(e)}")
                # 继续执行，不中断可视化过程
            
            # 基站：绘制 Docking Station 并标注序号
            for station_idx, station in enumerate(solution, start=1):
                try:
                    x, y = station
                    # 获取基站高度 - 改进的错误处理
                    try:
                        dist, ind = self.bc_kdtree.query(np.array([[x, y]]), return_distance=False)
                        roof_altitude = self.building_candidates[ind[0][0], 2]
                        z = roof_altitude
                    except Exception as e:
                        logger.warning(f"计算基站{station_idx}高度时出错: {str(e)}")
                        # 如果无法获取屋顶高度，使用地形高度加上默认建筑高度
                        try:
                            z = self.terrain.get_elevation(x, y) + 20.0  # 默认建筑高度20米
                        except:
                            z = 20.0  # 最后的后备值
                    
                    # 计算当前子图的行列位置
                    current_row = idx // cols + 1
                    current_col = idx % cols + 1
                    
                    # 添加基站点
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x],
                            y=[y],
                            z=[z],
                            mode='markers+text',
                            marker=dict(size=10, color='red', symbol='diamond-open'),
                            text=[f"Station {station_idx}"],  # 使用英文避免字体问题
                            textposition="top center",
                            name=f"{algo_name} Station {station_idx}",  # 使用英文避免字体问题
                            showlegend=(station_idx == 1)  # 只显示第一个基站的图例
                        ),
                        row=current_row,
                        col=current_col
                    )
                    
                    # 添加覆盖范围圆
                    try:
                        theta = np.linspace(0, 2 * np.pi, 50)
                        circle_x = list(x + Config.UAV_COVERAGE_RADIUS * np.cos(theta))
                        circle_y = list(y + Config.UAV_COVERAGE_RADIUS * np.sin(theta))
                        circle_z = [z] * len(theta)
                        fig.add_trace(
                            go.Scatter3d(
                                x=circle_x,
                                y=circle_y,
                                z=circle_z,
                                mode='lines',
                                line=dict(color='red', dash='dot'),
                                name='Coverage Range',  # 使用英文避免字体问题
                                showlegend=False
                            ),
                            row=current_row,
                            col=current_col
                        )
                    except Exception as e:
                        logger.warning(f"绘制基站{station_idx}覆盖范围时出错: {str(e)}")
                except Exception as e:
                    logger.warning(f"处理基站{station_idx}时出错: {str(e)}")
            
            # 添加覆盖率和适应度信息
            try:
                # 计算当前子图的行列位置
                current_row = idx // cols + 1
                current_col = idx % cols + 1
                fig.add_annotation(
                    x=0.5, y=0.95,
                    text=f"Coverage: {coverage:.2f}% | Fitness: {result['fitness']:.2f}",  # 使用英文避免字体问题
                    showarrow=False,
                    xref=f"x{current_row}{current_col} domain",
                    yref=f"y{current_row}{current_col} domain",
                    font=dict(size=14, color="red")
                    # 调整文本位置已在上方设置
                )
            except Exception as e:
                logger.warning(f"添加算法{algo_name}注释信息时出错: {str(e)}")
        
        # 更新布局
        fig.update_layout(
            title="Algorithm Solution Comparison",  # 使用英文避免字体问题
            height=900,
            width=1200,  # 固定宽度，适合2x2布局
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        # 更新每个场景的视角
        for i in range(1, min(len(self.results), 4) + 1):
            row_idx = (i-1) // cols + 1
            col_idx = (i-1) % cols + 1
            fig.update_scenes(
                xaxis_title="UTM X (m)",
                yaxis_title="UTM Y (m)",
                zaxis_title="Elevation (m)",  # 使用英文避免字体问题
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),  # 增加z轴比例，使地形不那么扁平
                row=row_idx, col=col_idx
            )
        
        # 保存为HTML文件
        try:
            fig.write_html(html_filename)
            logger.info(f"3D可视化结果已保存至: {html_filename}")
            return fig
        except Exception as e:
            logger.error(f"保存3D可视化HTML文件时出错: {str(e)}")
            # 尝试使用简化的布局重新保存
            try:
                # 简化布局设置
                fig.update_layout(height=600, width=200 * len(self.results))
                # 确保输出文件夹存在
                os.makedirs('../visualization_outputs', exist_ok=True)
                if not html_filename.startswith('../'):
                    html_filename = f"../visualization_outputs/{html_filename}"
                fig.write_html(html_filename)
                logger.info(f"简化版3D可视化结果已保存至: {html_filename}")
                return fig
            except Exception as e2:
                logger.error(f"保存简化版3D可视化也失败: {str(e2)}")
                return None

# ======================== 主函数 ========================
def run_algorithm_comparison(tiff_path, buildings_path, roads_path, algorithms=None, output_path="algorithm_comparison_results.png", html_path="algorithm_comparison_3d.html", save_results=True, results_file="algorithm_results.json"):
   
    # 加载地形
    terrain = TiffTerrainLoader(tiff_path)
    # 加载建筑物和路网
    try:
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Point, LineString
        from test import get_building_candidates, load_buildings, load_roads, generate_task_points
    except ImportError:
        logger.error("缺少geopandas或相关依赖模块，请安装后重试。")
        return
    
    # 加载建筑物数据
    buildings = None
    try:
        if os.path.exists(buildings_path):
            buildings = load_buildings(buildings_path, terrain)
        else:
            logger.error(f"建筑物数据文件不存在: {buildings_path}")
            return {}
    except Exception as e:
        logger.error(f"加载建筑物数据失败: {str(e)}")
        return {}
    
    # 检查建筑物数据是否有效
    if buildings is None or len(buildings) == 0:
        logger.error("建筑物数据为空，无法继续执行算法比较。")
        return {}
    
    # 提取建筑物候选点
    building_candidates = get_building_candidates(buildings, terrain)
    if building_candidates is None or len(building_candidates) == 0:
        logger.error("未提取到任何建筑物候选点，算法流程终止。")
        return
    
    # 加载道路数据
    roads = None
    try:
        if os.path.exists(roads_path):
            roads = load_roads(roads_path, terrain)
        else:
            logger.error(f"道路数据文件不存在: {roads_path}")
            return {}
    except Exception as e:
        logger.error(f"加载道路数据失败: {str(e)}")
        return {}
    
    # 检查道路数据是否有效
    if roads is None or len(roads) == 0:
        logger.error("道路数据为空，无法继续执行算法比较。")
        return {}
    
    # 生成任务点
    task_points = generate_task_points(roads, Config.NUM_TASK_POINTS)
    if task_points is None or len(task_points) == 0:
        logger.error("任务点生成失败，算法流程终止。")
        return
    # 算法比较
    comparison = AlgorithmComparison(terrain, building_candidates, task_points, buildings, roads)
    results = comparison.run_comparison(algorithms)
    # 绘制比较图表
    comparison.plot_comparison(output_path)
    
    # 生成3D可视化
    comparison.visualize_solutions(html_path)
    
    # 保存结果到文件
    if save_results:
        comparison.save_results(results_file)
    
    # 输出结果
    for algo_name, result in comparison.results.items():
        logger.info(f"{algo_name}算法结果: 适应度={result['fitness']:.2f}, 覆盖率={result['coverage']:.2f}%, 耗时={result['time']:.2f}秒")
    
    return results

# 如果直接运行此脚本
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行算法比较实验')
    parser.add_argument('--tiff', type=str, default=r"C:\Users\17370\Desktop\工作\山区无人机竞赛筹备\地理数据\投影后1\地形数据\Extract_广州de1.tif", help='地形数据TIFF文件路径')
    parser.add_argument('--buildings', type=str, default=r"C:\Users\17370\Desktop\工作\山区无人机竞赛筹备\地理数据\投影后1\建筑物\建筑物.shp", help='建筑物数据SHP文件路径')
    parser.add_argument('--roads', type=str, default=r"C:\Users\17370\Desktop\工作\山区无人机竞赛筹备\地理数据\投影后1\路网\路网.shp", help='道路数据SHP文件路径')
    parser.add_argument('--algorithms', type=str, nargs='+', help='要运行的算法列表，不指定则运行所有算法')
    parser.add_argument('--output', type=str, default="algorithm_comparison_results.png", help='2D比较图表保存路径')
    parser.add_argument('--html', type=str, default="algorithm_comparison_3d.html", help='3D可视化HTML文件保存路径')
    parser.add_argument('--results', type=str, default="algorithm_results.json", help='算法结果JSON文件保存路径')
    parser.add_argument('--no-save', action='store_true', help='不保存算法结果到文件')
    
    args = parser.parse_args()
    
    # 运行比较实验
    run_algorithm_comparison(
        args.tiff, 
        args.buildings, 
        args.roads, 
        algorithms=args.algorithms,
        output_path=args.output,
        html_path=args.html,
        save_results=not args.no_save,
        results_file=args.results
    )