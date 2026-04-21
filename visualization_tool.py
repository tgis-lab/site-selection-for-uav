#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KDTree

# 设置matplotlib字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16  # 四号字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入现有系统的模块
from test import Config, TiffTerrainLoader

# ======================== 日志配置 ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ======================== 可视化工具类 ========================
class AlgorithmVisualization:
    """算法结果可视化工具，从保存的结果文件中读取数据并进行可视化"""
    
    def __init__(self, results_file=None, terrain=None, building_candidates=None, task_points=None, roads=None, buildings=None):
        """初始化可视化工具
        
        Args:
            results_file: 算法结果JSON文件路径
            terrain: 地形数据对象，如果为None则尝试从结果文件中加载
            building_candidates: 建筑物候选点数组，如果为None则尝试从结果文件中加载
            task_points: 任务点数组，如果为None则尝试从结果文件中加载
            roads: 路网数据，如果为None则尝试从结果文件中加载
            buildings: 建筑物数据，如果为None则尝试从结果文件中加载
        """
        self.results_file = results_file
        self.terrain = terrain
        self.building_candidates = building_candidates
        self.task_points = task_points
        self.roads = roads
        self.buildings = buildings
        self.results = {}
        self.bc_kdtree = None

        # 如果提供了结果文件路径，则尝试加载
        if self.results_file:
            self._load_results()

        # 如果外部未提供地形和点数据，尝试从结果文件中加载
        if self.terrain is None and self.results_file and 'terrain_path' in self.metadata:
            try:
                self.terrain = TiffTerrainLoader(self.metadata['terrain_path'])
                logger.info(f"从结果文件加载地形数据: {self.metadata['terrain_path']}")
            except Exception as e:
                logger.error(f"加载地形数据失败: {str(e)}")

        if self.building_candidates is None and self.results_file and 'building_candidates' in self.metadata:
            self.building_candidates = np.array(self.metadata['building_candidates'])
            logger.info(f"从结果文件加载建筑物候选点: {len(self.building_candidates)}个")

        if self.task_points is None and self.results_file and 'task_points' in self.metadata:
            self.task_points = np.array(self.metadata['task_points'])
            logger.info(f"从结果文件加载任务点: {len(self.task_points)}个")

        # 初始化KDTree
        if self.building_candidates is not None:
            self.bc_kdtree = KDTree(self.building_candidates[:, :2])
    
    def _load_results(self):
        """从JSON文件加载算法结果"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                self.results = data['results']
                self.metadata = data['metadata']
                logger.info(f"成功加载算法结果，共{len(self.results)}个算法")
                
                # 记录路网和建筑物数据路径（如果有）
                if 'roads_path' not in self.metadata and 'road_path' in self.metadata:
                    self.metadata['roads_path'] = self.metadata['road_path']
                
                if 'buildings_path' not in self.metadata and 'building_path' in self.metadata:
                    self.metadata['buildings_path'] = self.metadata['building_path']
        except Exception as e:
            logger.error(f"加载结果文件失败: {str(e)}")
            self.results = {}
            self.metadata = {}
    
    def plot_comparison(self, save_path="algorithm_visualization_results.png"):
        """绘制算法比较图表，包含数据标准化处理和丰富的图表类型"""
        if not self.results:
            logger.warning("没有可用的比较结果")
            return None
        
        try:
            # 提取比较数据
            algo_names = list(self.results.keys())
            coverages = [self.results[algo]['coverage'] for algo in algo_names]
            fitnesses = [self.results[algo]['fitness'] for algo in algo_names]
            times = [self.results[algo]['time'] for algo in algo_names]
            
            # 数据标准化处理
            # 1. 计算各指标的最大值和最小值
            max_fitness = max(fitnesses) if fitnesses else 1.0
            min_fitness = min(fitnesses) if fitnesses else 0.0
            max_coverage = 100  # 覆盖率最大为100%
            min_coverage = 0    # 覆盖率最小为0%
            max_time = max(times) if times else 1.0
            min_time = min(times) if times else 0.1
            
            # 2. 标准化数据到[0,1]区间
            norm_fitnesses = [(f - min_fitness) / (max_fitness - min_fitness) if max_fitness != min_fitness else 0.5 for f in fitnesses]
            norm_coverages = [c / 100 for c in coverages]  # 覆盖率已经是百分比
            # 时间取倒数标准化，因为时间越短越好
            norm_time_efficiency = [1 - ((t - min_time) / (max_time - min_time)) if max_time != min_time else 0.5 for t in times]
            
            # 创建图表 - 使用2x2布局
            fig = plt.figure(figsize=(20, 16))
            
            # 1. 适应度柱状图 (左上)
            ax1 = fig.add_subplot(2, 2, 1)
            # 使用标准化后的颜色渐变
            colors1 = plt.cm.Blues(np.array(norm_fitnesses))
            bars1 = ax1.bar(algo_names, fitnesses, color=colors1, width=0.6)
            ax1.set_title('Algorithm Fitness Comparison', fontsize=18, fontweight='bold')
            ax1.set_ylabel('Fitness Value', fontsize=16)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax1.tick_params(axis='both', labelsize=16)
            # 在柱状图上添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_fitness,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=16)
            
            # 2. 覆盖率柱状图 (右上)
            ax2 = fig.add_subplot(2, 2, 2)
            # 使用标准化后的颜色渐变
            colors2 = plt.cm.Greens(np.array(norm_coverages))
            bars2 = ax2.bar(algo_names, coverages, color=colors2, width=0.6)
            ax2.set_title('Algorithm Coverage Comparison', fontsize=18, fontweight='bold')
            ax2.set_ylabel('Coverage (%)', fontsize=16)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.tick_params(axis='both', labelsize=16)
            # 在柱状图上添加数值标签
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=16)
            
            # 3. 热力图 - 综合性能比较 (左下)
            ax3 = fig.add_subplot(2, 2, 3)
            # 准备热力图数据
            heatmap_data = np.array([norm_fitnesses, norm_coverages, norm_time_efficiency])
            im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
            
            # 设置热力图标签
            ax3.set_yticks(np.arange(3))
            ax3.set_yticklabels(['Fitness', 'Coverage', 'Time Efficiency'], fontsize=16)
            ax3.set_xticks(np.arange(len(algo_names)))
            ax3.set_xticklabels(algo_names, fontsize=16)
            
            # 添加热力图数值标签
            for i in range(3):
                for j in range(len(algo_names)):
                    if i == 0:
                        text = f'{fitnesses[j]:.2f}'
                    elif i == 1:
                        text = f'{coverages[j]:.2f}%'
                    else:
                        text = f'{times[j]:.2f}s'
                    ax3.text(j, i, text, ha="center", va="center", color="white" if heatmap_data[i, j] < 0.7 else "black", fontsize=16)
            
            ax3.set_title('Algorithm Performance Heatmap', fontsize=18, fontweight='bold')
            fig.colorbar(im, ax=ax3, label='Normalized Performance')
            
            # 4. 雷达图 - 综合比较 (右下)
            ax4 = fig.add_subplot(2, 2, 4, polar=True)
            
            # 设置雷达图的角度
            categories = ['Fitness', 'Coverage', 'Time Efficiency']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合雷达图
            
            # 绘制雷达图
            ax4.set_theta_offset(np.pi / 2)  # 从顶部开始
            ax4.set_theta_direction(-1)  # 顺时针
            
            # 设置雷达图的刻度标签
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories, fontsize=16)
            
            # 绘制每个算法的雷达图
            for i, algo in enumerate(algo_names):
                values = [norm_fitnesses[i], norm_coverages[i], norm_time_efficiency[i]]
                values += values[:1]  # 闭合雷达图
                ax4.plot(angles, values, linewidth=2, label=algo)
                ax4.fill(angles, values, alpha=0.25)
            
            ax4.set_title('Algorithm Performance Radar Chart', fontsize=18, fontweight='bold')
            # 将图例放在图表下方，避免遮挡
            ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=16)
            
            # 添加标题和调整布局
            plt.suptitle('Multi-dimensional Algorithm Performance Analysis', fontsize=22, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"增强版比较结果图表已保存至: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制比较图表时出错: {str(e)}")
            # 尝试使用简化版本的图表
            try:
                # 创建简化版图表
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                
                # 覆盖率对比
                bars1 = ax1.bar(algo_names, coverages, color='skyblue')
                ax1.set_title('Coverage Comparison (%)', fontsize=18, fontweight='bold')
                ax1.set_ylim(0, 100)
                ax1.set_ylabel('Coverage (%)', fontsize=16)
                ax1.tick_params(axis='both', labelsize=16)
                ax1.set_xticklabels(algo_names, rotation=45, ha='right', fontsize=16)
                
                # 适应度对比
                bars2 = ax2.bar(algo_names, fitnesses, color='lightgreen')
                ax2.set_title('Fitness Comparison', fontsize=18, fontweight='bold')
                ax2.set_ylabel('Fitness', fontsize=16)
                ax2.tick_params(axis='both', labelsize=16)
                ax2.set_xticklabels(algo_names, rotation=45, ha='right', fontsize=16)
                
                # 执行时间对比
                bars3 = ax3.bar(algo_names, times, color='salmon')
                ax3.set_title('Execution Time Comparison (s)', fontsize=18, fontweight='bold')
                ax3.set_ylabel('Execution Time (s)', fontsize=16)
                ax3.tick_params(axis='both', labelsize=16)
                ax3.set_xticklabels(algo_names, rotation=45, ha='right', fontsize=16)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"简化版比较结果图表已保存至: {save_path}")
                
                return fig
            except Exception as e2:
                logger.error(f"绘制简化版图表也失败: {str(e2)}")
                return None
    
    def visualize_solutions(self, html_filename="algorithm_comparison_3d.html"):
        """在3D环境中可视化不同算法的解决方案，使用2x2方形布局"""
        if not self.results:
            logger.warning("没有可用的比较结果")
            return None
        
        # 检查必要的数据是否可用
        if self.terrain is None or self.task_points is None or self.building_candidates is None:
            logger.error("缺少必要的地形或点数据，无法生成3D可视化")
            # 尝试从metadata中加载数据
            try:
                if 'terrain_path' in self.metadata and self.terrain is None:
                    self.terrain = TiffTerrainLoader(self.metadata['terrain_path'])
                    logger.info(f"重新加载地形数据: {self.metadata['terrain_path']}")
                
                if 'building_candidates' in self.metadata and self.building_candidates is None:
                    self.building_candidates = np.array(self.metadata['building_candidates'])
                    self.bc_kdtree = KDTree(self.building_candidates[:, :2])
                    logger.info(f"重新加载建筑物候选点: {len(self.building_candidates)}个")
                
                if 'task_points' in self.metadata and self.task_points is None:
                    self.task_points = np.array(self.metadata['task_points'])
                    logger.info(f"重新加载任务点: {len(self.task_points)}个")
                
                # 再次检查数据
                if self.terrain is None or self.task_points is None or self.building_candidates is None:
                    logger.error("无法加载必要的地形或点数据，3D可视化失败")
                    return None
            except Exception as e:
                logger.error(f"加载可视化数据时出错: {str(e)}")
                return None
        
        # 尝试从metadata中加载路网和建筑物数据
        roads = None
        buildings = None
        try:
            if 'roads_path' in self.metadata:
                from test import load_roads
                roads = load_roads(self.metadata['roads_path'], self.terrain)
                logger.info(f"从结果文件加载路网数据: {self.metadata['roads_path']}")
            
            if 'buildings_path' in self.metadata:
                from test import load_buildings
                buildings = load_buildings(self.metadata['buildings_path'], self.terrain)
                logger.info(f"从结果文件加载建筑物数据: {self.metadata['buildings_path']}")
        except Exception as e:
            logger.warning(f"加载路网或建筑物数据时出错: {str(e)}")
            # 继续执行，不中断可视化过程
        
        # 计算布局的行列数 - 根据算法数量动态调整布局
        num_algos = len(self.results)
        
        # 根据算法数量优化布局
        if num_algos <= 2:
            rows, cols = 1, 2  # 1行2列布局
        else:
            rows, cols = 2, 2  # 2行2列布局
        
        if num_algos > 4:
            logger.warning(f"算法数量({num_algos})超过4个，只显示前4个算法")
            algo_items = list(self.results.items())[:4]
        else:
            algo_items = list(self.results.items())
        
        # 创建子图 - 使用优化后的布局
        subplot_titles = [algo_name for algo_name, _ in algo_items]
        specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=specs,
            horizontal_spacing=0.08,  # 增加水平间距
            vertical_spacing=0.15     # 增加垂直间距
        )
        
        # 为每个算法创建一个场景
        for idx, (algo_name, result) in enumerate(algo_items):
            # 计算在2x2网格中的位置
            row = idx // cols + 1  # 行号从1开始
            col = idx % cols + 1   # 列号从1开始
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
                    colorbar=dict(title='海拔 (m)', x=-0.07) if idx == 0 else None
                ),
                row=row, col=col
            )
            
            # 添加建筑物（如果有）
            if buildings is not None and not buildings.empty:
                try:
                    # 创建DEM多边形用于筛选建筑物
                    dem_poly = Polygon([
                        (float(np.min(self.terrain.utm_x)), float(np.min(self.terrain.utm_y))),
                        (float(np.max(self.terrain.utm_x)), float(np.min(self.terrain.utm_y))),
                        (float(np.max(self.terrain.utm_x)), float(np.max(self.terrain.utm_y))),
                        (float(np.min(self.terrain.utm_x)), float(np.max(self.terrain.utm_y)))
                    ])
                    valid_buildings = buildings[buildings.centroid.within(dem_poly)]
                    
                    # 限制显示的建筑物数量，避免过多导致性能问题
                    max_buildings = getattr(Config, 'MAX_BUILDINGS_TO_PLOT', 500)
                    if max_buildings is not None and len(valid_buildings) > max_buildings:
                        valid_buildings = valid_buildings.sample(max_buildings)
                        logger.info(f"限制显示建筑物数量为: {max_buildings}")
                    
                    for idx, row in valid_buildings.iterrows():
                        geom = row.geometry
                        if geom is None:
                            continue
                        if geom.geom_type in ["Polygon", "MultiPolygon"]:
                            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                            for poly in polys:
                                xs, ys = poly.exterior.xy
                                xs_list = list(xs)
                                ys_list = list(ys)
                                valid_pts = [(x, y) for x, y in zip(xs_list, ys_list) if x is not None and y is not None]
                                if valid_pts:
                                    xs_filtered, ys_filtered = zip(*valid_pts)
                                    zs_filtered = [self.terrain.get_elevation(x, y) for x, y in zip(xs_filtered, ys_filtered)]
                                    fig.add_trace(
                                        go.Scatter3d(
                                            x=list(xs_filtered),
                                            y=list(ys_filtered),
                                            z=zs_filtered,
                                            mode='lines',
                                            line=dict(color='gray', width=2),
                                            name='建筑物轮廓',
                                            showlegend=(idx == 0 and row == 1 and col == 1)  # 只在第一个子图显示一次图例
                                        ),
                                        row=row, col=col
                                    )
                except Exception as e:
                    logger.warning(f"绘制建筑物时出错: {str(e)}")
            
            # 添加路网（如果有）
            if roads is not None and not roads.empty:
                try:
                    all_road_x, all_road_y, all_road_z = [], [], []
                    for geom in roads.geometry:
                        if geom is None:
                            continue
                        if geom.geom_type == "LineString":
                            xs, ys = geom.xy
                            xs = list(xs)
                            ys = list(ys)
                            zs = [self.terrain.get_elevation(x, y) + 2 for x, y in zip(xs, ys)]  # 路网高出地面2米
                            all_road_x.extend(xs + [None])
                            all_road_y.extend(ys + [None])
                            all_road_z.extend(zs + [None])
                        elif geom.geom_type == "MultiLineString":
                            for line in geom.geoms:
                                xs, ys = line.xy
                                xs = list(xs)
                                ys = list(ys)
                                zs = [self.terrain.get_elevation(x, y) + 2 for x, y in zip(xs, ys)]
                                all_road_x.extend(xs + [None])
                                all_road_y.extend(ys + [None])
                                all_road_z.extend(zs + [None])
                    
                    if all_road_x:  # 确保有路网数据
                        fig.add_trace(
                            go.Scatter3d(
                                x=all_road_x,
                                y=all_road_y,
                                z=all_road_z,
                                mode='lines',
                                line=dict(color='gray', width=3),
                                name='路网',
                                opacity=0.8,
                                showlegend=(idx == 0)  # 只在第一个子图显示图例
                            ),
                            row=row, col=col
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
                        name='任务点',
                        opacity=0.8,
                        showlegend=(idx == 0)  # 只在第一个子图显示图例
                    ),
                    row=row, col=col
                )
            except Exception as e:
                logger.warning(f"绘制任务点时出错: {str(e)}")
                # 继续执行，不中断可视化过程
            
            # 基站：绘制 Docking Station 并标注序号
            for station_idx, station in enumerate(solution, start=1):
                try:
                    x, y = station
                    # 获取基站高度
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
                
                # 添加基站点 - 改进文本位置避免重叠
                # 根据基站索引动态调整文本位置，更好地分散标签
                text_positions = ["top center", "top right", "bottom left", "bottom right", "top left", "bottom center"]
                # 使用基站索引和算法索引的组合来选择不同的文本位置，确保更好的分散性
                position_idx = (station_idx + idx) % len(text_positions)
                text_position = text_positions[position_idx]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='markers+text',
                        marker=dict(size=15, color='red', symbol='diamond-open', line=dict(width=2, color='black')),  # 增大标记尺寸并添加黑色边框
                        text=[f"基站 {station_idx}"],
                        textposition=text_position,
                        textfont=dict(size=16, color="black", family="Arial Bold"),  # 增大字体并使用黑色加粗字体
                        name=f"{algo_name}基站 {station_idx}",
                        showlegend=(station_idx == 1)  # 只显示第一个基站的图例
                    ),
                    row=current_row, col=current_col
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
                            name='覆盖范围',
                            showlegend=False
                        ),
                        row=current_row, col=current_col
                    )
                except Exception as e:
                    logger.warning(f"绘制基站{station_idx}覆盖范围时出错: {str(e)}")
                    # 继续执行，不中断可视化过程
            
            # 添加覆盖率和适应度信息 - 优化位置和字体大小
            try:
                # 计算当前子图的行列位置
                current_row = idx // cols + 1
                current_col = idx % cols + 1
                # 将注释位置移到更低的位置，避免与标题重叠
                fig.add_annotation(
                    x=0.5, y=0.85,  # 将y从0.95调整为0.85，避免与标题重叠
                    text=f"覆盖率: {coverage:.2f}% | 适应度: {result['fitness']:.2f}",
                    showarrow=False,
                    xref=f"x{current_row}{current_col} domain",
                    yref=f"y{current_row}{current_col} domain",
                    font=dict(size=16, color="#0066cc", family="Arial"),  # 改变字体颜色为蓝色
                    bgcolor="rgba(255,255,255,0.8)",  # 增加背景不透明度
                    bordercolor="#0066cc",  # 边框颜色与字体匹配
                    borderwidth=1,
                    borderpad=4
                )
            except Exception as e:
                logger.warning(f"添加算法{algo_name}注释信息时出错: {str(e)}")
                # 继续执行，不中断可视化过程
        
        # 更新布局 - 优化高度和宽度比例以充分利用页面空间
        fig.update_layout(
            title={
                'text': "算法解决方案比较",  # 简化标题
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=32, family='Arial')
            },
            height=1000,  # 增加高度以更好地利用垂直空间
            width=max(1200, 400 * len(self.results)),  # 动态调整宽度
            margin=dict(l=10, r=10, b=10, t=80),  # 增加顶部边距，为标题留出更多空间
            legend=dict(
                orientation="h",  # 水平图例布局
                yanchor="bottom",
                y=-0.1,  # 将图例放在图表下方
                xanchor="center",
                x=0.5,
                font=dict(size=16),
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        # 更新每个场景的视角 - 优化3D视角参数
        for i in range(1, len(self.results) + 1):
            row_idx = (i-1)//cols+1
            col_idx = (i-1)%cols+1
            fig.update_scenes(
                xaxis_title="UTM X (m)",
                yaxis_title="UTM Y (m)",
                zaxis_title="海拔 (m)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.6),  # 增加z轴比例，使地形更加立体
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.0),  # 优化视角角度
                    up=dict(x=0, y=0, z=1)  # 确保正确的上方向
                ),
                xaxis=dict(showspikes=False),  # 移除坐标轴刺线
                yaxis=dict(showspikes=False),
                zaxis=dict(showspikes=False),
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
                fig.update_layout(height=700, width=250 * len(self.results))
                fig.write_html(html_filename)
                logger.info(f"简化版3D可视化结果已保存至: {html_filename}")
                return fig
            except Exception as e2:
                logger.error(f"保存简化版3D可视化也失败: {str(e2)}")
                return None

# ======================== 主函数 ========================
def visualize_from_file(results_file, output_path="../visualization_outputs/visualization_results.png", html_path="../visualization_outputs/visualization_3d.html"):
    """从保存的结果文件中加载数据并进行可视化
    
    Args:
        results_file: 算法结果JSON文件路径
        output_path: 2D比较图表保存路径
        html_path: 3D可视化HTML文件保存路径
    
    Returns:
        可视化结果字典
    """
    try:
        # 检查结果文件是否存在
        if not os.path.exists(results_file):
            logger.error(f"结果文件不存在: {results_file}")
            return None
            
        # 初始化可视化工具
        visualizer = AlgorithmVisualization(results_file)
        
        if not visualizer.results:
            logger.error(f"结果文件中没有有效的算法结果数据: {results_file}")
            return None
        
        # 绘制比较图表
        logger.info("正在生成算法比较图表...")
        visualizer.plot_comparison(output_path)
        
        # 生成3D可视化
        logger.info("正在生成3D可视化...")
        visualizer.visualize_solutions(html_path)
        
        # 输出结果
        logger.info("算法比较结果:")
        for algo_name, result in visualizer.results.items():
            logger.info(f"{algo_name}算法结果: 适应度={result['fitness']:.2f}, 覆盖率={result['coverage']:.2f}%, 耗时={result['time']:.2f}秒")
        
        logger.info(f"可视化完成! 2D图表: {output_path}, 3D可视化: {html_path}")
        return visualizer.results
    except Exception as e:
        logger.error(f"可视化过程中发生错误: {str(e)}")
        return None

# 如果直接运行此脚本
class AlgorithmVisualization:
    def __init__(self, results_file=None):
        self.results_file = results_file
        self.results = None
        self.metadata = {}

    def visualize_comparison2_results(self, results, baseline, save_path="comparison2_visualization.png"):
        """专门用于comparison2的可视化函数，展示各算法与基准线的区别"""
        try:
            algo_names = [r['algorithm'] for r in results]
            fitness_scores = [r['fitness'] for r in results]
            normalized_scores = [r['normalized_fitness'] for r in results]

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))

            # 绘制适应度柱状图
            bars = ax.bar(algo_names, fitness_scores, color='skyblue', label='Fitness Scores')

            # 绘制基准线
            ax.axhline(y=baseline['max_fitness'], color='red', linestyle='--', label='Max Fitness Baseline')
            ax.axhline(y=baseline['average_fitness'], color='green', linestyle='--', label='Average Fitness Baseline')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.2f}', ha='center', va='bottom')

            # 设置标题和标签
            ax.set_title('Comparison2 Algorithm Performance', fontsize=16, fontweight='bold')
            ax.set_ylabel('Fitness Scores', fontsize=14)
            ax.set_xlabel('Algorithms', fontsize=14)
            ax.legend(fontsize=12)

            # 保存图表
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            logger.info(f"comparison2可视化图表已保存至: {save_path}")

            return fig
        except Exception as e:
            logger.error(f"绘制comparison2可视化图表时出错: {str(e)}")