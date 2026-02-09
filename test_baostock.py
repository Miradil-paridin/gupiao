# 测试 baostock 是否正常
import baostock as bs

lg = bs.login()
print(f"登录状态: {lg.error_code} - {lg.error_msg}")

rs = bs.query_history_k_data_plus(
    "sh.600519",  # 茅台
    "date,open,high,low,close,volume",
    start_date="2024-01-01",
    end_date="2024-01-10",
    frequency="d",
    adjustflag="2"
)

print(f"查询状态: {rs.error_code} - {rs.error_msg}")

data = []
while rs.next():
    data.append(rs.get_row_data())

print(f"数据行数: {len(data)}")
print(data[:3])

bs.logout()