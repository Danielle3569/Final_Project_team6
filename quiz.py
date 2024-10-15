import os
from text_split import split_into_sentences,join_sentences
from gpt import chatgpt_response

filename = '사회문화-A단원-자연 현상과 사회문화 현상의 비교-개념강의.txt'

text_file = open(filename,'r',encoding='utf-8')
text = text_file.read()

max_token = 800

sentences = split_into_sentences(text)
split_text = join_sentences(sentences,max_token)

prompt = 'I want you to read a transcript of a college lecture, create a multiple choice question based on it, and write the solution. In multiple choice question, you must named "### 문제" in question, and "### 정답 및 해설" in answer. And be sure to answer in Korean. '

list_result = []
for source in split_text:
    result = chatgpt_response(prompt,source,max_token)
    list_result.append(result)

list_quiz = []
list_answer = []
for text in list_result:
    quiz = text.split("###")[1]
    list_quiz.append(quiz)
    answer = text.split("###")[2]
    list_answer.append(answer)