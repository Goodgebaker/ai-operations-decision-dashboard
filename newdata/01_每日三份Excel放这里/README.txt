每天把同一日期的三份文件放到这个目录，然后双击项目根目录的“更新每日数据.bat”。

文件名必须保持以下格式：
1. 模型性能中间明细_YYYYMMDD.xlsx
2. 模型性能忙时对比_YYYYMMDD.xlsx
3. NPU中间统计表_YYYYMMDD.xlsx

程序会先校验三份文件，再脱敏实例/IP、追加历史数据并重建看板。
成功处理的原始文件会移入 newdata/archive/YYYYMMDD/，不会提交到 Git。
