#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import math
import numpy as np
import rasterio
from rasterio.transform import Affine
from pyproj import Proj, Transformer
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import map_coordinates
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import webbrowser

# ======================== 全局配置 ========================
class Config:
    # UAV及系统参数
    UAV_HEIGHT = 100.0              # 用于地形z值计算 (m)
    MIN_SAFE_ALTITUDE = 50.0        # 最低安全飞行高度 (m)
    UAV_DEPLOYMENT_ALTITUDE = 60.0  # UAV出动时固定高度 (m)
    UAV_COVERAGE_RADIUS = 5000    # UAV覆盖半径 (m)
    UAV_SPEED = 18.0                # UAV飞行速度 (m/s)
    P_m = 10.0                    # UAV运动功率 (W)
    BATTERY_CAPACITY = 400000.0     # 电池容量 (J)
    ENERGY_PENALTY_FACTOR = 0.001   # 能量超支惩罚系数

    # 优化参数
    NUM_STATIONS = 10           # Docking Station 数量
    PSO_PARTICLES = 40            # PSO种群数量 (灵敏度分析推荐)
    MAX_ITERATIONS = 100          # 最大迭代次数 (灵敏度分析推荐)
    
    # 自适应PSO参数 (优化后)
    PSO_INERTIA_MAX = 0.95        # 最大惯性权重 (提高初始探索)
    PSO_INERTIA_MIN = 0.3         # 最小惯性权重 (降低最终值增强开发)
    PSO_C1_INITIAL = 2.0          # 初始认知学习因子 (温和调整)
    PSO_C2_INITIAL = 0.8          # 初始社会学习因子 (提高初始社会学习)
    PSO_C1_FINAL = 0.8            # 最终认知学习因子 (保持一定个体经验)
    PSO_C2_FINAL = 2.2            # 最终社会学习因子 (适度增强群体协作)
    
    # 收敛检测参数 (放宽条件)
    CONVERGENCE_THRESHOLD = 1e-4  # 收敛阈值 (放宽收敛条件)
    STAGNATION_LIMIT = 15         # 停滞迭代限制 (增加容忍度)
    
    # 多样性维护参数
    MIN_DIVERSITY_THRESHOLD = 0.1  # 最小多样性阈值
    MUTATION_RATE = 0.15          # 变异率
    MUTATION_STRENGTH = 0.3       # 变异强度
    
    NUM_TASK_POINTS = 100         # 任务点数量

    # 可视化参数（全部显示建筑物，不限制数量）
    MAX_BUILDINGS_TO_PLOT = None

    # 地理参数
    UTM_ZONE = 49                 # UTM带号

# ======================== 日志配置 ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ======================== 地形数据加载模块 ========================
class TiffTerrainLoader:
    def __init__(self, tiff_path: str):
        self.tiff_path = tiff_path
        self.original_elevation = None
        self.elevation = None
        self.transform = None
        self.crs = None
        self._load_tiff()
        self._preprocess_terrain()

    def _load_tiff(self):
        with rasterio.open(self.tiff_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            def _is_transform_valid(t: Affine) -> bool:
                params = [t.a, t.b, t.c, t.d, t.e, t.f]
                return not any(math.isnan(p) or math.isinf(p) for p in params) and abs(t.determinant) > 1e-6
            if not _is_transform_valid(self.transform):
                raise ValueError(f"非法仿射变换参数: {self.transform.to_gdal()}")
            raw_data = src.read(1)
            scale = getattr(src, 'scales', [1.0])[0]
            offset = getattr(src, 'offsets', [0.0])[0]
            no_data = src.nodata if src.nodata is not None else -32768
            self.original_elevation = raw_data.astype(np.float32) * scale + offset
            valid_mask = (self.original_elevation != (no_data * scale + offset))
            self.original_elevation[~valid_mask] = np.nan
            self.original_elevation = np.nan_to_num(self.original_elevation, nan=np.nanmedian(self.original_elevation))
            self.elevation = self.original_elevation.copy()
            self._generate_coords()
            logger.info(f"DEM加载成功 | 尺寸: {self.elevation.shape}")
            logger.info(f"高程范围: {np.min(self.original_elevation)} ~ {np.max(self.original_elevation)}")

    def enhance_terrain(self, vertical_exaggeration: float = 3.0):
        if self.original_elevation is None:
            raise AttributeError("原始高程数据未初始化")
        self.elevation = self.original_elevation * vertical_exaggeration
        self._generate_coords()
        self._preprocess_terrain()
        logger.info(f"地形增强完成 | 新平均坡度: {np.mean(self.slope):.1f}°")

    def _generate_coords(self):
        rows, cols = self.elevation.shape
        x_coords = np.arange(cols) * self.transform.a + self.transform.c
        y_coords = np.arange(rows) * self.transform.e + self.transform.f
        self.utm_x, self.utm_y = np.meshgrid(x_coords, y_coords)
        logger.info(f"UTM坐标生成: X[{self.utm_x.min():.1f}, {self.utm_x.max():.1f}]m, Y[{self.utm_y.min():.1f}, {self.utm_y.max():.1f}]m")

    def _preprocess_terrain(self):
        dx, dy = np.gradient(self.elevation)
        self.slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        self.ground_z = self.elevation
        self.z_values = self.elevation + Config.UAV_HEIGHT
        logger.info(f"地形预处理完成 | 平均坡度: {np.mean(self.slope):.1f}°")

    def query_elevation(self, points: np.ndarray) -> np.ndarray:
        if not isinstance(points, np.ndarray) or points.shape[1] != 2:
            raise ValueError("输入坐标应为Nx2的numpy数组")
        rows, cols = [], []
        for x, y in points:
            col, row = ~self.transform * (x, y)
            rows.append(row)
            cols.append(col)
        rows = np.array(rows)
        cols = np.array(cols)
        rows = np.clip(rows, 0, self.elevation.shape[0]-1)
        cols = np.clip(cols, 0, self.elevation.shape[1]-1)
        return map_coordinates(self.elevation, [rows, cols], order=1, mode='nearest')

    def get_elevation(self, x, y):
        if x < np.min(self.utm_x) or x > np.max(self.utm_x) or y < np.min(self.utm_y) or y > np.max(self.utm_y):
            return float(np.mean(self.elevation))
        else:
            return float(self.query_elevation(np.array([[x, y]]))[0])

# ======================== 建筑物候选点提取 ========================
def get_building_candidates(buildings: gpd.GeoDataFrame, terrain: TiffTerrainLoader) -> np.ndarray:
    """
    对每个建筑物，使用 representative_point() 保证点在建筑内部，
    并计算屋顶高度：roof_altitude = terrain.get_elevation(rep.x, rep.y) + Height (若存在，否则默认20m)。
    返回候选点数组，形状为 (M, 3)，每行：[x, y, roof_altitude]
    """
    candidates = []
    for idx, row in buildings.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        rep = geom.representative_point()
        x, y = rep.x, rep.y
        base = terrain.get_elevation(x, y)
        height = 20.0
        if 'Height' in buildings.columns:
            try:
                h_val = float(row['Height'])
                if h_val > 0:
                    height = h_val
            except Exception:
                pass
        roof_altitude = base + height
        candidates.append([x, y, roof_altitude])
    candidates = np.array(candidates)
    logger.info(f"提取建筑物候选点数量: {len(candidates)}")
    return candidates

# ======================== 数据加载辅助函数 ========================
def load_roads(shp_path: str, terrain: TiffTerrainLoader) -> list:
    try:
        roads_gdf = gpd.read_file(shp_path)
        if roads_gdf.crs is None:
            logger.warning("路网数据缺失CRS，默认设置为EPSG:4326")
            roads_gdf.set_crs(epsg=4326, inplace=True)
        if roads_gdf.crs != terrain.crs:
            roads_gdf = roads_gdf.to_crs(terrain.crs)
        roads = []
        for geom in roads_gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == "MultiLineString":
                roads.extend(list(geom.geoms))
            elif geom.geom_type == "LineString":
                roads.append(geom)
        logger.info(f"加载路网成功，路段数: {len(roads)}")
        return roads
    except Exception as e:
        logger.error(f"加载路网数据失败: {str(e)}")
        return []

def load_buildings(shp_path: str, terrain: TiffTerrainLoader) -> gpd.GeoDataFrame:
    try:
        bld_gdf = gpd.read_file(shp_path)
        if bld_gdf.crs is None:
            logger.warning("建筑物数据缺失CRS，默认设置为EPSG:4326")
            bld_gdf.set_crs(epsg=4326, inplace=True)
        if bld_gdf.crs != terrain.crs:
            bld_gdf = bld_gdf.to_crs(terrain.crs)
        logger.info(f"加载建筑物数据成功，记录数: {len(bld_gdf)}")
        return bld_gdf
    except Exception as e:
        logger.error(f"加载建筑物数据失败: {str(e)}")
        return gpd.GeoDataFrame()

def generate_task_points(roads: list, num_points: int) -> np.ndarray:
    task_points = []
    if not roads:
        logger.warning("路网数据为空，无法生成任务点。")
        return np.empty((0,2))
    for _ in range(num_points):
        road = np.random.choice(roads)
        rand_dist = np.random.uniform(0, road.length)
        point = road.interpolate(rand_dist)
        task_points.append([point.x, point.y])
    task_points = np.array(task_points)
    logger.info(f"生成任务点成功，数量: {len(task_points)}")
    return task_points

# ======================== 统一适应度函数 ========================
def _compute_los_penalty_piecewise(broken: int, num_stations: int, 
                                    threshold1: float = 0.05, penalty1: float = 200.0,
                                    threshold2: float = 0.20, penalty2_slope: float = 30.0 / 0.15,
                                    threshold3: float = 0.50, penalty3_slope: float = 40.0 / 0.30,
                                    penalty4_slope: float = 20.0 / 0.50) -> float:
    """分段化视线（LoS）惩罚函数

    参数:
        broken: 基站对中不可见（断线）的数量。
        num_stations: 基站数量，用于计算成对组合的总数。
        threshold1: 第一个分段阈值（轻度遮挡）。
        penalty1: 第一个分段的惩罚系数。
        threshold2: 第二个分段阈值（中度遮挡）。
        penalty2_slope: 第二个分段的惩罚斜率。
        threshold3: 第三个分段阈值（重度遮挡）。
        penalty3_slope: 第三个分段的惩罚斜率。
        penalty4_slope: 第四个分段的惩罚斜率。

    返回:
        惩罚值，范围在 [0, 100] 之间，随不可见比例单调递增并在高遮挡时饱和。

    说明:
        将不可见对的占比分段线性映射为惩罚值，避免阶跃式大惩罚导致总分失真。
        分段阈值可根据场景调整。
    """
    total_pairs = max(num_stations * (num_stations - 1) // 2, 1)
    frac = broken / total_pairs

    if frac <= threshold1:
        penalty = penalty1 * frac
    elif frac <= threshold2:
        penalty = (penalty1 * threshold1) + (frac - threshold1) * penalty2_slope
    elif frac <= threshold3:
        penalty = (penalty1 * threshold1) + ((threshold2 - threshold1) * penalty2_slope) + (frac - threshold2) * penalty3_slope
    else:
        penalty = (penalty1 * threshold1) + ((threshold2 - threshold1) * penalty2_slope) + ((threshold3 - threshold2) * penalty3_slope) + (frac - threshold3) * penalty4_slope

    return max(0.0, min(100.0, float(penalty)))

# ======================== PSO优化器模块 ========================
class DronePSOOptimizer:
    building_candidates = None
    bc_kdtree = None

    def __init__(self, terrain, building_candidates: np.ndarray, task_points: np.ndarray):
        self.terrain = terrain
        self.task_points = task_points
        self.building_candidates = building_candidates  # shape (M,3)
        self.bc_kdtree = KDTree(building_candidates[:, :2])
        DronePSOOptimizer.building_candidates = building_candidates
        DronePSOOptimizer.bc_kdtree = self.bc_kdtree
        self.num_particles = Config.PSO_PARTICLES
        self.num_stations = Config.NUM_STATIONS
        self.dim = 2
        self.particles = self._init_particles()
        self.velocities = np.zeros_like(self.particles)
        self.pbest = self.particles.copy()
        self.pbest_scores = np.full(self.num_particles, -np.inf)
        self.gbest = None
        self.gbest_score = -np.inf
        
        # 自适应参数
        self.current_iteration = 0
        self.max_iterations = Config.MAX_ITERATIONS
        self.inertia_weight = Config.PSO_INERTIA_MAX
        self.c1 = Config.PSO_C1_INITIAL
        self.c2 = Config.PSO_C2_INITIAL
        
        # 收敛检测
        self.fitness_history = []
        self.stagnation_count = 0
        self.best_fitness_history = []
        
        # 多样性度量
        self.diversity_history = []
        
        # 新增LoS惩罚参数初始化
        self.los_penalty_params = None
        
        logger.info(f"PSO优化器初始化完成 - 粒子数: {self.num_particles}, 停靠站数: {self.num_stations}")

    def _init_particles(self):
        particles = np.empty((self.num_particles, self.num_stations, self.dim))
        num_candidates = self.building_candidates.shape[0]
        for i in range(self.num_particles):
            indices = np.random.choice(num_candidates, self.num_stations, replace=True)
            particles[i] = self.building_candidates[indices, :2]
        return particles

    def project_to_candidate(self, p):
        dist, ind = self.bc_kdtree.query(np.array([p]), return_distance=True)
        return self.building_candidates[ind[0][0], :2]

    def get_docking_altitude(self, p):
        dist, ind = self.bc_kdtree.query(np.array([p]), return_distance=True)
        return self.building_candidates[ind[0][0], 2]

    def has_line_of_sight(self, p1, p2):
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

    def compute_fitness(self, solution, los_penalty_params=None):
        """统一适应度函数，考虑覆盖率、能量消耗、分段化视线阻断惩罚和运行时间

        参数:
            solution: 基站位置数组，形状为 (num_stations, 2)。
            los_penalty_params: 包含LoS惩罚函数参数的字典，例如：
                                {'threshold1': 0.05, 'penalty1': 200.0, ...}

        返回:
            fitness: 原始适应度分数（覆盖为正、能耗与LoS为负）。

        说明:
            - 覆盖率按任务点在覆盖半径内的比例映射到百分制。
            - 能耗以超过电池容量的比例加权惩罚。
            - LoS 惩罚采用分段线性映射，范围限定在 [0,100]，避免过度惩罚。
        """
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
            for j in range(i + 1, n):
                if not self.has_line_of_sight(solution[i], solution[j]):
                    broken += 1
        
        used_params = los_penalty_params or self.los_penalty_params
        if used_params:
            los_penalty = _compute_los_penalty_piecewise(broken, n, **used_params)
        else:
            los_penalty = _compute_los_penalty_piecewise(broken, n)

        # 总适应度
        fitness = coverage_ratio - energy_penalty - los_penalty
        return fitness

    def _update_adaptive_parameters(self):
        """自适应更新PSO参数"""
        progress = self.current_iteration / Config.MAX_ITERATIONS
        
        # 改进的非线性惯性权重递减策略（指数递减）
        base_inertia = Config.PSO_INERTIA_MAX * (Config.PSO_INERTIA_MIN / Config.PSO_INERTIA_MAX) ** progress
        
        # 温和的学习因子调整
        base_c1 = Config.PSO_C1_INITIAL - (Config.PSO_C1_INITIAL - Config.PSO_C1_FINAL) * progress
        base_c2 = Config.PSO_C2_INITIAL + (Config.PSO_C2_FINAL - Config.PSO_C2_INITIAL) * progress
        
        # 计算当前多样性
        current_diversity = self._calculate_diversity()
        self.diversity_history.append(current_diversity)
        
        # 基于多样性的自适应调整（更温和的调整策略）
        diversity_factor = 1.0
        if current_diversity < Config.MIN_DIVERSITY_THRESHOLD:
            # 多样性过低，增加探索
            diversity_factor = 1.0 + 0.15 * (Config.MIN_DIVERSITY_THRESHOLD - current_diversity)
        elif current_diversity > 0.4:
            # 多样性过高，促进收敛
            diversity_factor = 1.0 - 0.1 * (current_diversity - 0.4)
        
        # 应用多样性调整
        self.inertia_weight = np.clip(base_inertia * diversity_factor, 
                                      Config.PSO_INERTIA_MIN, Config.PSO_INERTIA_MAX)
        
        # 学习因子的温和调整
        c1_adjustment = 1.0
        c2_adjustment = 1.0
        
        if current_diversity < Config.MIN_DIVERSITY_THRESHOLD:
            # 增强个体学习，减弱社会学习
            c1_adjustment = 1.0 + 0.1 * (Config.MIN_DIVERSITY_THRESHOLD - current_diversity)
            c2_adjustment = 1.0 - 0.05 * (Config.MIN_DIVERSITY_THRESHOLD - current_diversity)
        elif current_diversity > 0.4:
            # 减弱个体学习，增强社会学习
            c1_adjustment = 1.0 - 0.05 * (current_diversity - 0.4)
            c2_adjustment = 1.0 + 0.1 * (current_diversity - 0.4)
        
        self.c1 = np.clip(base_c1 * c1_adjustment, 0.5, 3.0)
        self.c2 = np.clip(base_c2 * c2_adjustment, 0.5, 3.0)
        
        # 基于收敛状态的调整（更保守的策略）
        if self.stagnation_count > 8:  # 提高阈值
            # 算法可能陷入局部最优，轻微增加探索
            if self.stagnation_count % 5 == 0:  # 降低调整频率
                self.inertia_weight = min(self.inertia_weight * 1.03, Config.PSO_INERTIA_MAX)
                self.c1 = min(self.c1 * 1.02, Config.PSO_C1_INITIAL)
    
    def _calculate_diversity(self):
        """计算种群多样性"""
        if self.num_particles <= 1:
            return 0.0
        
        # 计算所有粒子对之间的平均距离
        total_distance = 0.0
        count = 0
        
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                # 计算两个粒子之间的欧几里得距离
                dist = np.linalg.norm(self.particles[i] - self.particles[j])
                total_distance += dist
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        
        # 归一化多样性（相对于搜索空间的大小）
        search_space_size = np.linalg.norm(
            [np.max(self.building_candidates[:, 0]) - np.min(self.building_candidates[:, 0]),
             np.max(self.building_candidates[:, 1]) - np.min(self.building_candidates[:, 1])]
        )
        
        diversity = avg_distance / search_space_size if search_space_size > 0 else 0.0
        return diversity
    
    def _check_convergence(self):
        """检查算法是否收敛"""
        if len(self.best_fitness_history) < 2:
            return False
        
        # 检查最佳适应度是否停滞
        current_best = self.best_fitness_history[-1]
        previous_best = self.best_fitness_history[-2]
        
        improvement = abs(current_best - previous_best)
        
        if improvement < Config.CONVERGENCE_THRESHOLD:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        # 如果停滞次数超过限制，认为已收敛
        return self.stagnation_count >= Config.STAGNATION_LIMIT
    
    def _apply_mutation(self):
        """在收敛时应用变异操作增加多样性"""
        # 计算当前多样性
        current_diversity = self._calculate_diversity()
        
        # 根据多样性调整变异策略
        if current_diversity < Config.MIN_DIVERSITY_THRESHOLD:
            # 强变异：重新初始化更多粒子
            mutation_rate = Config.MUTATION_RATE * 2.0
        else:
            # 标准变异
            mutation_rate = Config.MUTATION_RATE
        
        num_mutate = max(1, int(self.num_particles * mutation_rate))
        
        # 选择适应度较差的粒子进行变异
        fitness_scores = [self.pbest_scores[i] for i in range(self.num_particles)]
        worst_indices = np.argsort(fitness_scores)[:num_mutate]
        
        for idx in worst_indices:
            # 使用变异强度参数控制变异程度
            if np.random.random() < Config.MUTATION_STRENGTH:
                # 完全重新初始化
                num_candidates = self.building_candidates.shape[0]
                indices = np.random.choice(num_candidates, self.num_stations, replace=True)
                self.particles[idx] = self.building_candidates[indices, :2]
                self.velocities[idx] = np.zeros_like(self.velocities[idx])
            else:
                # 部分变异：只改变部分停靠站位置
                num_mutate_stations = max(1, int(self.num_stations * 0.5))
                station_indices = np.random.choice(self.num_stations, num_mutate_stations, replace=False)
                
                for station_idx in station_indices:
                    num_candidates = self.building_candidates.shape[0]
                    new_candidate_idx = np.random.choice(num_candidates)
                    self.particles[idx, station_idx] = self.building_candidates[new_candidate_idx, :2]
        
        logger.info(f"应用变异操作，重新初始化 {num_mutate} 个粒子 (多样性: {current_diversity:.3f})")

    def optimize(self):
        """改进的PSO优化算法，包含自适应参数调整"""
        logger.info("开始自适应PSO优化...")
        
        for iteration in tqdm(range(Config.MAX_ITERATIONS), desc="自适应PSO迭代"):
            self.current_iteration = iteration
            
            # 更新自适应参数
            self._update_adaptive_parameters()
            
            # 评估所有粒子的适应度
            iteration_best_fitness = -np.inf
            for b in range(self.num_particles):
                fitness = self.compute_fitness(self.particles[b])
                
                # 更新个体最佳
                if fitness > self.pbest_scores[b]:
                    self.pbest[b] = self.particles[b].copy()
                    self.pbest_scores[b] = fitness
                
                # 更新全局最佳
                if fitness > self.gbest_score:
                    self.gbest = self.particles[b].copy()
                    self.gbest_score = fitness
                
                # 记录当前迭代最佳
                if fitness > iteration_best_fitness:
                    iteration_best_fitness = fitness
            
            # 记录历史信息
            self.best_fitness_history.append(self.gbest_score)
            self.fitness_history.append(iteration_best_fitness)
            
            # 更新粒子位置和速度
            for b in range(self.num_particles):
                # 生成随机因子
                r1 = np.random.uniform(0, 1, size=(self.num_stations, self.dim))
                r2 = np.random.uniform(0, 1, size=(self.num_stations, self.dim))
                
                # 更新速度（使用自适应参数）
                self.velocities[b] = (self.inertia_weight * self.velocities[b] +
                                      self.c1 * r1 * (self.pbest[b] - self.particles[b]) +
                                      self.c2 * r2 * (self.gbest - self.particles[b]))
                
                # 速度限制
                max_velocity = 1000.0  # 最大速度限制
                self.velocities[b] = np.clip(self.velocities[b], -max_velocity, max_velocity)
                
                # 更新位置
                self.particles[b] += self.velocities[b]
                
                # 投影到候选点
                for s in range(self.num_stations):
                    self.particles[b, s] = self.project_to_candidate(self.particles[b, s])
            
            # 检查收敛
            if self._check_convergence():
                logger.info(f"算法在第 {iteration + 1} 次迭代时收敛")
                # 应用变异操作尝试跳出局部最优
                if iteration < Config.MAX_ITERATIONS - 1:  # 不是最后一次迭代
                    self._apply_mutation()
                    self.stagnation_count = 0  # 重置停滞计数
            
            # 每10次迭代输出一次详细信息
            if (iteration + 1) % 10 == 0 or iteration == 0:
                diversity = self.diversity_history[-1] if self.diversity_history else 0
                logger.info(
                    f"迭代 {iteration + 1}/{Config.MAX_ITERATIONS}: "
                    f"最佳适应度={self.gbest_score:.2f}, "
                    f"惯性权重={self.inertia_weight:.3f}, "
                    f"c1={self.c1:.3f}, c2={self.c2:.3f}, "
                    f"多样性={diversity:.3f}"
                )
        
        logger.info(f"自适应PSO优化完成，全局最佳适应度: {self.gbest_score:.2f}")
        logger.info(f"最终参数 - 惯性权重: {self.inertia_weight:.3f}, c1: {self.c1:.3f}, c2: {self.c2:.3f}")
        
        return self.gbest, self.gbest_score

    def get_coverage_efficiency(self, particle):
        diffs = self.task_points[:, np.newaxis, :] - particle[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        covered = min_dists <= Config.UAV_COVERAGE_RADIUS
        return np.sum(covered) / len(self.task_points) * 100.0
    
    def get_optimization_statistics(self):
        """获取优化过程的统计信息"""
        return {
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history,
            'final_inertia_weight': self.inertia_weight,
            'final_c1': self.c1,
            'final_c2': self.c2,
            'convergence_iteration': len(self.best_fitness_history)
        }

# ======================== 主程序入口 ========================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler('deployment.log'), logging.StreamHandler()]
    )
    try:
        # 直接指定各数据的文件路径（请根据实际情况修改路径）
        terrain_path = r"C:\Users\17370\Desktop\工作\山区无人机竞赛筹备\地理数据\投影后1\地形数据\Extract_广州de1.tif"
        building_shp_path = r"C:\Users\17370\Desktop\工作\山区无人机竞赛筹备\地理数据\投影后1\建筑物\建筑物.shp"
        road_shp_path = r"C:\Users\17370\Desktop\工作\山区无人机竞赛筹备\地理数据\投影后1\路网\路网.shp"

        logger.info("正在加载地形数据...")
        terrain = TiffTerrainLoader(terrain_path)
        # 如有需要，可调用 terrain.enhance_terrain(vertical_exaggeration=3.0)

        logger.info("加载建筑物数据...")
        buildings = load_buildings(building_shp_path, terrain)
        logger.info("加载路网数据...")
        roads = load_roads(road_shp_path, terrain)
        logger.info("生成任务点...")
        task_points = generate_task_points(roads, Config.NUM_TASK_POINTS)
        if task_points.size == 0:
            raise ValueError("任务点数组为空，请检查路网数据！")

        # 提取建筑物候选点（建筑物屋顶）
        building_candidates = get_building_candidates(buildings, terrain)
        logger.info("初始化PSO优化器...")
        optimizer = DronePSOOptimizer(terrain, building_candidates, task_points)
        logger.info("开始PSO优化...")
        best_solution, best_score = optimizer.optimize()
        coverage_eff = optimizer.get_coverage_efficiency(best_solution)
        logger.info(f"优化完成 | 最佳适应度: {best_score:.2f}  覆盖率: {coverage_eff:.2f}%")

        np.savetxt("best_solution.csv", best_solution, header="UTM_X,UTM_Y", delimiter=",")
        transformer = Transformer.from_proj(
            Proj(proj='utm', zone=Config.UTM_ZONE, ellps='WGS84'),
            Proj(proj='latlong')
        )
        lons, lats = transformer.transform(best_solution[:,0], best_solution[:,1])
        geo_coords = np.column_stack([lons, lats])
        np.savetxt("geo_coordinates.csv", geo_coords, header="Longitude,Latitude", delimiter=",")
        station_ids = np.arange(1, best_solution.shape[0] + 1)
        nest_geo_df = pd.DataFrame({
            'Nest_ID': station_ids,
            'UTM_X': best_solution[:, 0],
            'UTM_Y': best_solution[:, 1],
            'Longitude': lons,
            'Latitude': lats
        })
        nest_geo_df.to_csv("nest_geodetic_coordinates.csv", index=False, encoding='utf-8-sig')
        logger.info("机巢大地坐标已保存至: nest_geodetic_coordinates.csv")
        logger.info("\n%s", nest_geo_df[['Nest_ID', 'Longitude', 'Latitude']].to_string(index=False))
        gdf = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in best_solution],
            data={'UTM_X': best_solution[:,0], 'UTM_Y': best_solution[:,1]},
            crs=terrain.crs
        )
        gdf.to_file("deployment_points.shp")
        logger.info("结果文件已保存")

        if 'visualize_results' in globals():
            logger.info("生成并显示可视化结果...")
            visualize_results(terrain, roads, buildings, task_points, best_solution, coverage_eff)
        else:
            logger.warning("未找到 visualize_results 函数，已跳过可视化步骤。")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
