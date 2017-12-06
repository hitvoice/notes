import re
import jieba

# jieba.set_dictionary('resources/dict.big.txt')
# jieba.load_userdict('resources/user.dict.txt')
# jieba.enable_parallel()

'''
注意：POS用的分词（jieba.posseg.cut）和普通的分词结果有明显差别（通常效果更差），
使用需谨慎
for token in jieba.posseg.cut(line):
    print(token.word, token.flag)
'''

# simplest version
def tokenize(line):
    return ' '.join(jieba.cut(line))


def inverse_tokenize(line):
    return re.sub('(.)( |$)', lambda m: m.group(1), line)

# work with sklearn.feature_extraction.text.TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = list(
    """ \n\r\t.,。，、＇：∶；?‘’“”〝〞﹕︰﹔﹖﹑·…;！!´？！～—ˉ｜‖＂〃｀@﹫﹟#﹩$
    ﹠&﹪%*﹡﹢﹦﹤‐―﹨ˆ˜﹍+=<——_-\~()[]（）〈〉‹›﹛﹜『』〖〗［］《》{}「」【】""")
X_raw = [tokenize(line) for line in lines]
vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b", min_df=2, stop_words=stop_words)
X = vectorizer.fit_transform(X_raw)
