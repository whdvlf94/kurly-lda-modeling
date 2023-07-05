import re # 문자열 처리를 위한 정규표현식 패키지
import warnings
warnings.filterwarnings(action="ignore")

from mecab import MeCab
from tqdm import tqdm # 작업 프로세스 시각화
from gensim import corpora # 단어 빈도수 계산 패키지
from wordcloud import WordCloud # LDA 시각화 - 워드클라우드
import gensim # LDA 모델 활용 목적
import pyLDAvis.gensim_models # LDA 시각화용 패키지
import pandas as pd
import matplotlib.pyplot as plt

mecab = MeCab()

replace_list = pd.read_excel("./data/replace_list.xlsx")
stopword_list = pd.read_excel("./data/stopword_list.xlsx")
one_char_keyword = pd.read_excel("./data/one_char_list.xlsx")


def replace_word(review: str):
    """
    같은 의미의 단어를 하나의 단어로 통일하는 함수
    """
    for idx, value in enumerate(replace_list["before_replacement"]):
        try:
            # 치환할 단어가 있는 경우에만 데이터 치환 수행
            if value in review:
                review = review.replace(value, replace_list["after_replacement"][idx])
        except Exception as e:
            print(f"Error 발생 / 에러명: {e}")
    return review

def remove_stopword(tokens: list[str]):
    """
    불용어 제거 함수
    1. 텍스트마이닝에서 빈출되는 어휘이지만 사람들의 반응이나 의견과 거리가 먼 서술어나 조사를 제외
    2. 의미있는 값을 가지기 어려운 1글자 키워드 제거(단, 한글에서는 한 글자여도 의미있는 키워드가 있는 경우가 있으니 예외처리 진행)
    """
    review_removed_stopword = []
    for token in tokens:
        # 토큰의 글자 수가 2글자 이상인 경우
        if 1 < len(token):
            # 토큰이 불용어가 아닌 경우만 분석용 리뷰 데이터로 포함
            if token not in list(stopword_list["stopword"]):
                review_removed_stopword.append(token)
        # 토큰의 글자 수가 1글자인 경우
        else:
            # 1글자 키워드에 포함되는 경우만 분석용 리뷰 데이터로 포함
            if token in list(one_char_keyword["one_char_keyword"]):
                review_removed_stopword.append(token)
    return review_removed_stopword

def select_review(review_removed_stopword: list[str], min: int = 3, max: int = 15):
    """
    각 리뷰에서 추출된 명사의 개수 범위를 설정하는 함수
    min: int: 최소 토큰 개수(default=3)
    max: int: 최대 토큰 개수(default=15)
    """
    review_prep = []
    for tokens in review_removed_stopword:
        if min <= len(tokens) <= max:
            review_prep.append(tokens)
    return review_prep


def lda_modeling(review_prep: list[str], num_topics: int = 10, passes: int = 15):
    """
    LDA 모델링 학습: 리뷰 내 텍스트 정수 인코딩 및 빈도수 계산을 토대로 LDA 모델을 학습하는 함수
    num_topics: int: 추출하고 싶은 토픽의 개수(Hyperparameter)
    passes: int: corpus 기반의 LDA 모델의 학습 횟수 
    """
    # 단어 인코딩 및 빈도수 계산
    dictionary = corpora.Dictionary(review_prep)
    corpus = [dictionary.doc2bow(review) for review in review_prep]
    # LDA 모델 학습
    model = gensim.models.ldamodel.LdaModel(corpus, 
                                            num_topics = num_topics, 
                                            id2word = dictionary, 
                                            passes = passes)
    return model, corpus, dictionary

def print_topic_prop(topics: list[list[str]]):
    """
    토픽별 단어 구성 출력 함수
    """
    topic_values = []
    for topic in topics:
        print(topic)
        topic_value = topic[1]
        topic_values.append(topic_value)
    topic_prop = pd.DataFrame({"topic_num" : list(range(1, len(topics) + 1)), "word_prop": topic_values})
    topic_prop.to_excel("./result/topic_review.xlsx")

def lda_visualize(model, corpus, dictionary):
    """
    LDA 모델링 시각화 함수: LDA 시각화용 패키지
    """
    result_visualized = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    # 시각화 결과 저장
    RESULT_FILE = f"./result/lda_result_review.html"
    pyLDAvis.save_html(result_visualized, RESULT_FILE)

def lda_word_cloud(model):
    """
    LDA 모델링 시각호 함수: 워드 클라우드
    """
    wc = WordCloud(background_color="white", font_path="./SeoulNamsanM.ttf")# 워드클라우드

    plt.figure(figsize=(30,30))
    for t in range(model.num_topics):
        plt.subplot(5,4,t+1)
        x = dict(model.show_topic(t,200))
        im = wc.generate_from_frequencies(x)
        plt.imshow(im)
        plt.axis("off")
        plt.title("Topic #" + str(t))

    plt.savefig(f"./result/LDA_wordcloud.png", bbox_inches="tight")     # 이미지 저장    

def main(reviews: list[str]):
    
    #한글 외 텍스트 삭제
    reviews = list(map(lambda review: re.sub("[^가-힣 ]", "", review), reviews))

    # 단어 치환 작업
    review_replaced_list = []
    for review in tqdm(reviews):
        review_replaced = replace_word(review)
        review_replaced_list.append(review_replaced)

    # 리뷰 데이터 토큰화
    review_tokenized = list(map(lambda review: mecab.nouns(review), review_replaced_list))

    # 불용어 제거 작업
    review_removed_stopword = list(map(lambda tokens : remove_stopword(tokens), review_tokenized))

    # 리뷰 데이터 토큰 범위 설정
    review_prep = select_review(review_removed_stopword)

    # LDA 모델 학습
    model, corpus, dictionary = lda_modeling(review_prep)
    topics = model.print_topics(num_words = model.num_topics)
    print_topic_prop(topics)

    # LDA 모델링 시각화
    lda_visualize(model, corpus, dictionary) # pyLDAvis
    lda_word_cloud(model) # WordCloud

if __name__ == "__main__":
    FILE_NAME = "fresh_data" # fresh_data(신선식품) or proceed_data(가공식품)
    
    # 리뷰 데이터
    df = pd.read_excel(f"data/reviews/{FILE_NAME}.xlsx", sheet_name=0)
    dataset = df.dropna(axis=0) # axis = 0: 결측치 포함한 모든 행 제거
    reviews = dataset["리뷰 내용"]

    main(reviews)
    