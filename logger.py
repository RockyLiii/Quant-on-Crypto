import logging

def get_logger(config):
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config['output_path']}/log.txt"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger