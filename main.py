from typing import Dict
import time
from timeline import BaseTimeline
from Strategy import StatArbitrageStrategy
from backtest import run_backtest, analyze_results
from logger import get_logger
from parser_1 import get_args
from visualize import plot_value_curve, plot_time_series, plot_position_curve, plot_all_features, plot_capital_curve
from data_preprocess import preprocess_data
import yaml # type: ignore


class Timer:
    def __init__(self, name: str, logger):
        self.name = name
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} 耗时: {elapsed:.4f} 秒")


def main():
    # 解析命令行参数
    args = get_args()
    
    # 读取配置文件
    with open(args.conf_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 设置日志
    logger = get_logger(config)
    logger.info("开始加载配置...")
    
    try:
        with Timer("整体回测", logger):
            if config.get('preprocess_data', True):  # 可以通过配置控制是否执行预处理
                logger.info("开始预处理数据...")
                with Timer("数据预处理", logger):
                    preprocess_data(config, logger)
                    pass
                logger.info("数据预处理完成")
            logger.info("初始化策略...")
            start = time.time()
            strategy = StatArbitrageStrategy(params=config)
            logger.info(f"策略初始化耗时: {time.time() - start:.4f} 秒")
            
            # 添加时间戳打印
            last_update = time.time()
            last_timestamp = None
            processed_count = 0
            
            def progress_callback(timestamp, data_len):
                nonlocal last_update, last_timestamp, processed_count
                processed_count += 1
                current_time = time.time()
                
                # 每100个时间戳或5秒打印一次进度
                if processed_count % 100 == 0 or current_time - last_update >= 5:
                    if last_timestamp:
                        speed = (timestamp - last_timestamp) / (current_time - last_update)
                        progress = processed_count / data_len * 100
                        logger.info(f"进度: {progress:.2f}% ({processed_count}/{data_len})")
                        logger.info(f"处理速度: {speed:.2f} ticks/s")
                    
                    last_update = current_time
                    last_timestamp = timestamp
            
            logger.info("开始执行回测...")
            
            timeline = run_backtest(
                strategy=strategy,
                logger=logger,
                backtest_params=config,
                progress_callback=progress_callback
            )
            
            logger.info("回测完成")
            # 获取配置中的时间间隔
            d_t = 300000
            print(f"回测时间间隔: {d_t}")
            
            # 生成资产曲线图，传入d_t以还原时间戳
            output_dir = config.get('output_path', 'output')
            logger.info("生成资产曲线图...")
            start_time = config['backtest']['start_time']
            end_time = config['backtest']['end_time']
            plot_value_curve(timeline, output_dir, d_t=d_t,start_time = start_time, end_time = end_time)
            plot_capital_curve(timeline, output_dir, d_t=d_t,start_time = start_time, end_time = end_time)
            plot_position_curve(timeline, output_dir, d_t=d_t,start_time = start_time, end_time = end_time)
            logger.info(f"资产曲线已保存至: {output_dir}/equity_curve.png")

            logger.info("生成特征曲线图...")
            plot_all_features(timeline, strategy, output_dir, d_t=d_t,start_time = start_time, end_time = end_time)
            logger.info(f"特征曲线已保存至: {output_dir}/features/")
                        
            # 分析回测结果
            with Timer("结果分析", logger):
                metrics = analyze_results(timeline, strategy, config, logger)
                
            
    except Exception as e:
        logger.error(f"回测过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()