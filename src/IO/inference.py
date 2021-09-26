


import datetime
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from ..data.prepare import read_CLDAS_FORCING, read_SMAP
from ..data.preprocess import preprocess_smap_nrt, read_CLDAS_DD, read_SMAP_DD, preprocess_cldas_nrt
from ..data.slices import read_CLDAS_PREP, slices_inputs, make_inference_inputs

# 当天的日期。
local_time = time.localtime()
year = local_time.tm_year
month = local_time.tm_mon
day = local_time.tm_mday

# 获取3天前-13天前的起止日期。
begin_date = datetime.datetime.now() - datetime.timedelta(days = 12)
end_date = datetime.datetime.now() - datetime.timedelta(days = 3)

# 将待输入的数据，制作为逐日的数据，并存储起来到指定文件夹
read_SMAP(input_path='/hard/lilu/SMAP_L4/SMAP_L4/',
          out_path='/hard/lilu/SMAP_L4/SMAP_L4_DD/',
          begin_date=begin_date,
          end_date=end_date,
          lat_lower=22,
          lat_upper=33,
          lon_left=110,
          lon_right=123)

read_CLDAS_FORCING(input_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING/',
                   out_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/',
                   begin_date=begin_date, 
                   end_date=end_date, 
                   lat_lower=22, 
                   lat_upper=33, 
                   lon_left=110, 
                   lon_right=123)

# 读取逐日的数据，根据遗留下来的normalized参数，进行预处理，并存储到指定文件夹。
preprocess_cldas_nrt(input_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/',
                    out_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD_PREP/',
                    begin_date=begin_date,
                    end_date=end_date)

preprocess_smap_nrt(input_path='/hard/lilu/SMAP_L4/SMAP_L4_DD/',
                    out_path='/hard/lilu/SMAP_L4/SMAP_L4_DD_PREP/',
                    begin_date=begin_date,
                    end_date=end_date)

# 对于不同的区域制作输入数据
for task in np.arange(142):
    make_inference_inputs(input_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/',
                        out_path='/hard/lilu/4_HRSMPP/inference/predict/',
                        begin_date=begin_date, 
                        end_date=end_date,
                        task=task,
                        lat_lower=22, 
                            lat_upper=33, 
                            lon_left=110, 
                            lon_right=123)


# 加载存储的模型，不同区域的预测

# 拼接，输出预测的结果。
