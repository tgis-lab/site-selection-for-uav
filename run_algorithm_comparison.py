#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from algorithm_comparison2 import run_algorithm_comparison

# ======================== 日志配置 ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ======================== 命令行参数解析 ========================
def parse_args():
    parser = argparse.ArgumentParser(description='无人机基站部署算法比较工具')
    parser.add_argument('--tiff', type=str, default='guangzhou_dem.tif', help='地形数据文件路径')
    parser.add_argument('--buildings', type=str, default='buildings.shp', help='建筑物数据文件路径')
    parser.add_argument('--roads', type=str, default='roads.shp', help='道路数据文件路径')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        choices=['K-means', 'GA', 'NSGA-II', 'ACO', 'PSO', 'all'],
                        default=['all'], help='要比较的算法')
    parser.add_argument('--output', type=str, default='../visualization_outputs/algorithm_comparison_results.png', 
                        help='结果图表保存路径')
    parser.add_argument('--html', type=str, default='../visualization_outputs/algorithm_comparison_3d.html',
                        help='3D可视化结果保存路径')
    return parser.parse_args()

# ======================== 主函数 ========================
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查地形文件是否存在（必须）
    if not os.path.exists(args.tiff):
        logger.error(f"地形数据文件不存在: {args.tiff}")
        return 1
        
    # 检查建筑物和道路文件是否存在（必须）
    if not os.path.exists(args.buildings):
        logger.warning(f"建筑物数据文件不存在: {args.buildings}，将使用建筑物候选点代替")
    if not os.path.exists(args.roads):
        logger.error(f"道路数据文件不存在: {args.roads}")
        return 1
    
    # 确定要比较的算法
    algorithms = None  # None表示全部算法
    if 'all' not in args.algorithms:
        algorithms = args.algorithms
    
    logger.info("开始运行算法比较实验...")
    logger.info(f"地形数据: {args.tiff}")
    logger.info(f"建筑物数据: {args.buildings}")
    logger.info(f"道路数据: {args.roads}")
    logger.info(f"比较算法: {algorithms if algorithms else '全部'}")
    
    try:
        # 导入test模块中的Config类，修改参数
        from test import Config
        
        # 可以在这里修改配置参数
        # Config.NUM_STATIONS = 8  # 基站数量
        # Config.MAX_ITERATIONS = 50  # 迭代次数
        # Config.PSO_PARTICLES = 20  # 粒子数量
        
        # 运行比较实验
        results = run_algorithm_comparison(
            args.tiff, 
            args.buildings, 
            args.roads, 
            algorithms=algorithms,
            output_path=args.output,
            html_path=args.html
        )
        
        logger.info("算法比较实验完成!")
        logger.info(f"结果图表已保存至: {args.output}")
        logger.info(f"3D可视化结果已保存至: {args.html}")
        
        # 打印结果摘要
        logger.info("\n算法比较结果摘要:")
        for algo_name, result in results.items():
            logger.info(f"{algo_name}: 适应度={result['fitness']:.2f}, 覆盖率={result['coverage']:.2f}%, 耗时={result['time']:.2f}秒")
        
        return 0
    
    except Exception as e:
        logger.error(f"运行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())