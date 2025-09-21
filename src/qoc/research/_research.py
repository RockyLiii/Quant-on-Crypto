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
        

def plot_single_timeseries(
    data_dict: dict, 
    data_name: str, 
    column_name: str, 
    title: str, 
    ylabel: str, 
    dir: str,
    filename: str
) -> None:
    data = data_dict[data_name][column_name]
    plt.figure(figsize=(12, 6))
    
    # 处理Series类型
    if isinstance(data, pd.Series):
        plt.plot(data.index, data.values, label=data.name or ylabel)
    else:
        plt.plot(data.index, data.iloc[:, 0], label=data.columns[0] or ylabel)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir}/{filename}")
    plt.close()

def plot_multiple_timeseries(
    data_dict: dict, 
    data_x_column: list[list[str]], 
    title: str, 
    ylabel: str, 
    dir: str,
    filename: str,
    normalize: bool = False,
) -> None:
    """
    绘制多个时间序列数据在同一张图上
    
    Args:
        data_dict: 包含数据的字典
        data_x_column: 数据指定列表，每项为 [data_name, column_name] 形式
        normalize: 是否将每个序列除以其第一个值进行归一化
        title: 图表标题
        ylabel: Y轴标签
        dir: 保存目录
        filename: 文件名
    """
    plt.figure(figsize=(12, 6))
    
    for [data_name, column_name] in data_x_column:
        data = data_dict[data_name][column_name]
        
        # 处理Series类型
        if isinstance(data, pd.Series):
            values = data.values
            label = f"{data_name}_{column_name}"
            
            # 归一化处理
            if normalize and len(values) > 0 and values[0] != 0:
                first_value = values[0]
                values = values / first_value
                label += " (normalized)"
                
            plt.plot(data.index, values, label=label)
        # 处理DataFrame类型
        else:
            values = data.iloc[:, 0].values
            label = f"{data_name}_{column_name}"
            
            # 归一化处理
            if normalize and len(values) > 0 and values[0] != 0:
                first_value = values[0]
                values = values / first_value
                label += " (normalized)"
                
            plt.plot(data.index, values, label=label)
    
    # 调整标题和标签
    if normalize:
        title += " (Normalized)"
        ylabel = "Relative Change (First Value = 1.0)"
    
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
        data_dict=total_data_dict,
        data_name='current_values',
        column_name='Total',
        title='Asset Value',
        ylabel='Asset Value(USDT)',
        dir=output_dir,
        filename='value.png'
    )

    
    print("绘制BTC网格价格对比图表...")
    plot_multiple_timeseries(
        data_dict=total_data_dict,
        data_x_column=[
            ['base_prices', 'BTCUSDT'],
            ['upper_prices', 'BTCUSDT'],
            ['lower_prices', 'BTCUSDT']
        ],
        title='BTC grid',
        ylabel='BTC price(USDT)',
        dir=output_dir,
        filename='btc_grid_prices.png'
    )


    print("绘制DOGE网格价格对比图表...")
    plot_multiple_timeseries(
        data_dict=total_data_dict,
        data_x_column=[
            ['base_prices', 'DOGEUSDT'],
            ['upper_prices', 'DOGEUSDT'],
            ['lower_prices', 'DOGEUSDT']
        ],
        title='DOGE grid',
        ylabel='DOGE price(USDT)',
        dir=output_dir,
        filename='doge_grid_prices.png'
    )


    print("绘制多币种价格对比图表...")
    plot_multiple_timeseries(
        data_dict=total_data_dict,
        data_x_column=[
            ['base_prices', 'BTCUSDT'],
            ['base_prices', 'DOGEUSDT'],
        ], 
        title='Multiple Assets Price Comparison',
        ylabel='Relative Price',
        dir=output_dir,
        filename='multiple_assets_prices.png',
        normalize=True,
    )
