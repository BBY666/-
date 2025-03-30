import requests
import json

from ipdb import set_trace

def get_instructions(text):
    url = 'http://10.50.0.37:5001/chat'
    input_data = {
        'input': f'你是一名操作机器狗的专家。我将提供给你一段描述机器狗动作的文字。你的任务是提取必要的指令，例如"前进"、"后退"、"停止"、"左转"、"右转"、"伸懒腰"、"坐下"、"站起来"，如果文本中包含相关语义的话，将其转化为基本指令并提取动作的程度，例如前进多少米，向左转多少度，或向右转多少度。以JSON格式返回动作、速度和动作程度。如果没有指定程度，"degrees"字段应为0。\
        例如：\
        - 文字："前进一米" -> `{{"action": "前进", "degrees": "1"}}`\
        - 文字："后退一米" -> `{{"action": "后退", "degrees": "1"}}`\
        - 文字："停止" -> 返回：`{{"action": "停止", "degrees": "0"}}`\
        - 文字："左转三十度" -> `{{"action": "左转", "degrees": "30"}}`\
        - 文字："右转三十度" -> `{{"action": "右转", "degrees": "30"}}`\
        - 文字："站起来" -> `{{"action": "站起来", "degrees": "0"}}`\
        - 文字："趴下" -> `{{"action": "趴下", "degrees": "0"}}`\
        如果文字中不存在包含的指令。\
        例如：\
        - 文字：你在做什么->`{{"action": "无", "degrees": "0"}}`\
        以下是我提供的文字：{text}'
    }
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(input_data))
        print(response)
        print(response.json()['response'])
        parsed_response = json.loads(response.json()['response'].replace('json\n', "").replace("`", ""))
        print(parsed_response)
        return parsed_response
    except (ValueError, KeyError, json.JSONDecodeError):
        return {"action": "无", "degrees": "0"}

if  __name__ == "__main__":
    text_result= get_instructions("前进一米")
    print(text_result)
