import datetime
import pandas as pd
import re
s= ["00010101:00:00:00.000000", "00010101:00:01:00.000000","01010101:10:00:00.000000"]


def date_parse(s):
    if type(s) == datetime.datetime:
        return s
    try:
        return pd.datetime.strptime(s, '%Y%m%d:%H:%M:%S.%f')
    except ValueError:
        return "Error"


pattern = re.compile(r'\d{8}:\d{2}:\d{2}:\d{2}.\d{6}')
result = pattern.match(s[1])
if result:
    print(1)
else:
    print(0)
