import argparse


# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="Run with YAML config")
    parser.add_argument(
        "--conf_path",
        type=str,
        default="config/example.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    return args
