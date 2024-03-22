import os
import numpy as np
# 定义一个函数，接受一个文件夹路径作为参数，返回该文件夹中所有txt文件，且文件名包括time.txt的列表
def find_txt_files_with_time(folder):
  # 定义一个空列表，用于存储符合条件的文件
  result = []
  # 遍历文件夹中的所有文件和子文件夹
  for file in os.listdir(folder):
    # 拼接文件的完整路径
    file_path = os.path.join(folder, file)
    # 判断文件是否是txt文件，且文件名包括time.txt
    if file_path.endswith(".txt") and "time.txt" in file_path:
      # 将文件路径添加到结果列表中
      result.append(file_path)
    # 判断文件是否是文件夹
    elif os.path.isdir(file_path):
      # 递归调用函数，将子文件夹中的符合条件的文件也添加到结果列表中
      result.extend(find_txt_files_with_time(file_path))
  # 返回结果列表
  return result


result =[]  # 每个序列求平均值
speed =[]  # 总帧数/总时间
frames = []
# 调用函数，传入你想要搜索的文件夹路径，例如"C:\\Users\\user\\Documents"
files = find_txt_files_with_time("/media/SSDPA/yanmiao/rgb/SeqTrack/test/tracking_results/ptb-tir/speed_interval_100/swa_seqtrack/mixformer_baseline_got")
# 打印结果列表


for i in files:
    with open(i, "r") as f:
        # 定义一个空列表，用于存储文件中的每行数值
        data = []
        # 遍历文件中的每一行
        for line in f:
            # 去掉行尾的换行符
            line = line.strip()
            # 判断行是否为空
            if line:
                # 将行转换为浮点数
                value = float(line)
                # 将数值添加到列表中
                data.append(value)
    # 将列表转换为numpy数组
    array = np.array(data)
    result.append(array.mean())
    frames.append(array.shape)
    speed.append(array.sum())

print(1/np.array(result).mean())
print(np.array(frames).sum()/np.array(speed).sum())
