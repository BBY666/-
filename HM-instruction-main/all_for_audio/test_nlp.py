import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# 定义模式
patterns = {
    "停": "停止|停下来|别动|不要|不",
    "前": "向前|前进|往前|走前面|前",
    "后": "向后|后退|往后|走后面|后",
    "左": "向左|左转|往左|走左边|左",
    "右": "向右|右转|往右|走右边|右",
}


def map_sentence_to_command(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    for command, pattern in patterns.items():
        for keyword in pattern.split("|"):
            if keyword in sentence:
                return command

    return "未识别的指令"


# 测试函数
sentence1 = "向前走"
sentence2 = "往后面走两三步"
sentence3 = "不要往前"

print(map_sentence_to_command(sentence1))  # 输出：前
print(map_sentence_to_command(sentence2))  # 输出：后
print(map_sentence_to_command(sentence3))  # 输出：停


#nltk.download("averaged_perceptron_tagger")
#nltk.download("punkt")