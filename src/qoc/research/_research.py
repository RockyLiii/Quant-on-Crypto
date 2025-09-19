import arcticdb as adb
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


def import_data(
    strategy_lib_path = '/Users/lizeyu/Desktop/Quant-on-Crypto/examples/grid-strategy/data/database/',
):
    uri = f"lmdb://{strategy_lib_path}"


    ac = adb.Arctic(uri)

    strategy_library = ac.get_library('strategy')

    libraries = ac.list_libraries()
    print(f"数据库中的所有库: {libraries}")


    symbols = strategy_library.list_symbols()
    print(f"策略库中的所有符号: {symbols}")



    total_data_dict = {}
    for symbol in symbols:
        print(f"\n指标名: {symbol} ")

        # 读取数据
        data = strategy_library.read(symbol).data
        
        # 转换为pandas DataFrame以便统一显示
        if not isinstance(data, pd.DataFrame):
            if hasattr(data, 'to_pandas'):
                data = data.to_pandas()
            else:
                print(f"无法转换为pandas DataFrame，原始数据:\n{data}")
                continue
        
        # 打印数据基本信息
        print(f"数据对象: {list(data.columns)}")
        # 显示基本信
        print(f"数据形状: {data.shape}")
        print(f"数据范围: {data.index.min()} 到 {data.index.max()}")
        total_data_dict[symbol] = data
    return total_data_dict
        

def plot_single_timeseries(data: pd.DataFrame, title: str, ylabel: str, dir: str,filename: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.iloc[:, 0], label=ylabel)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir}/{filename}")
    plt.close()


if __name__ == "__main__":
    
    total_data_dict = import_data()

    output_dir = '/Users/lizeyu/Desktop/Quant-on-Crypto/examples/grid-strategy/output'

    print("绘制资产价值图表...")
    plot_single_timeseries(
        data=total_data_dict['current_values'],
        title='Asset Value',
        ylabel='Asset Value(USDT)',
        dir=output_dir,
        filename='value.png'
    )

