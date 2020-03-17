import re
from collections import Counter


# 单词切分
def words(text):
    return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('big.txt').read()))


# 出现一个词的概率
def P(word, N=sum(WORDS.values())):
    return WORDS[word]/N


# 预测可能性最大的词
def correction(word):
    return max(candidates(word), key=P)


# 返回四组单词，首选是已知，其次是之差一个单词的，再其次是差两个单词
def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    return set(w for w in words if w in WORDS)


# 和已知单词仅有一个字母之差
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


# 和已知单词仅有两个字母之差
def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


if __name__ == '__main__':
    print(correction('chian'))
    