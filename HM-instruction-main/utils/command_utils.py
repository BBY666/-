import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
import re

# 定义模式
patterns = {
    "停止": "停止|停下来|别动|不要|不",
    "前进": "向前|前进|往前|走前面|前",
    "后退": "向后|后退|往后|走后面|后",
    "向左": "向左|往左|走左边|左",
    "向右": "向右|往右|走右边|右",
    "右转": "转右圈|右转一圈|右旋转|右绕圈|右转",
    "左转": "转左圈|左转一圈|左旋转|左绕圈|左转",
    "坐下": "坐下|坐下来|坐着|坐下去|趴下|坐|趴",
    "站起来": "站起来|起立|站起|站起来|站",
    "伸懒腰": "伸懒腰|伸展|拉伸|舒展|懒"
}

# 提取语句中的命令
def map_sentence_to_command(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    for command, pattern in patterns.items():
        for keyword in pattern.split("|"):
            if keyword in sentence:
                return command

    return "未识别的指令"

# 将中文数字转化为阿拉伯数字
def chinese_to_arabic(chinese_num):
    chinese_digits = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9
    }
    
    chinese_units = {
        '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000
    }
    
    stack = []
    result = 0
    temp_unit = 1  # 当前的单位值
    temp_num = 0   # 当前的数字值

    for char in chinese_num:
        if char in chinese_digits:
            if len(stack) == 0: 
                stack.append(chinese_digits[char]) 
            else:
                temp_num = stack.pop() 
                result += temp_num * temp_unit
                temp_unit = 1
                stack.append(chinese_digits[char])
        elif char in chinese_units: # 如果是后面chinese units，就是要乘以倍数
            unit_value = temp_unit * chinese_units[char] # 倍数
            temp_unit = unit_value
            
            
            # else:
            #     temp_unit = unit_value
            #     result += temp_num * temp_unit
            #     temp_num = 0
    if len(stack) != 0:
        result = result + stack.pop() * temp_unit
    
    return result

# 从语句中提取中文数字
def extract_chinese_numbers(text):
    # 定义匹配连续中文数字的正则表达式
    pattern = r'[零一二三四五六七八九十百千万亿]+'
    
    # 使用正则表达式查找所有连续的中文数字段
    chinese_numbers = re.findall(pattern, text)
    
    # 返回匹配到的中文数字列表
    return chinese_numbers

# 示例语句
# text = "今天我买了五十二个苹果，总共花了一千零一元。"
# result = extract_chinese_numbers(text)
# print("提取出的中文数字:", result)
# print(chinese_to_arabic("一千零一"))
        