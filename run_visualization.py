#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='无人机基站部署算法运行与可视化工具')
    subparsers = parser.add_subparsers(dest='command', help='选择运行模式')
    
    # 算法运行子命令
    run_parser = subparsers.add_parser('run', help='运行算法并保存结果')
    run_parser.add_argument('--tiff', type=str, required=True, help='地形数据TIFF文件路径')
    run_parser.add_argument('--buildings', type=str, required=True, help='建筑物数据SHP文件路径')
    run_parser.add_argument('--roads', type=str, required=True, help='道路数据SHP文件路径')
    run_parser.add_argument('--algorithms', type=str, nargs='+', help='要运行的算法列表，不指定则运行所有算法')
    run_parser.add_argument('--output', type=str, default="algorithm_comparison_results.png", help='2D比较图表保存路径')
    run_parser.add_argument('--html', type=str, default="algorithm_comparison_3d.html", help='3D可视化HTML文件保存路径')
    run_parser.add_argument('--results', type=str, default="algorithm_results.json", help='算法结果JSON文件保存路径')
    
    # 可视化子命令
    vis_parser = subparsers.add_parser('visualize', help='从保存的结果文件中可视化算法结果')
    vis_parser.add_argument('--results', type=str, required=True, help='算法结果JSON文件路径')
    vis_parser.add_argument('--output', type=str, default="../visualization_outputs/visualization_results.png", help='2D比较图表保存路径')
    vis_parser.add_argument('--html', type=str, default="../visualization_outputs/visualization_3d.html", help='3D可视化HTML文件保存路径')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # 运行算法并保存结果
        from algorithm_comparison2 import run_algorithm_comparison
        logger.info("正在运行算法比较实验...")
        run_algorithm_comparison(
            args.tiff, 
            args.buildings, 
            args.roads, 
            algorithms=args.algorithms,
            output_path=args.output,
            html_path=args.html,
            save_results=True,
            results_file=args.results
        )
        logger.info(f"算法运行完成，结果已保存至 {args.results}")
        
    elif args.command == 'visualize':
        # 从保存的结果文件中可视化
        from visualization_tool import visualize_from_file
        logger.info(f"正在从 {args.results} 加载算法结果并进行可视化...")
        visualize_from_file(args.results, args.output, args.html)
        logger.info("可视化完成")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()