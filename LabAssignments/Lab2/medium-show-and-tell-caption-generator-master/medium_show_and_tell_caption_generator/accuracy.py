import nltk
from itertools import zip_longest
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from PyRouge.pyrouge import Rouge

r = Rouge()
list = []


def accuracy():
    with open("C:\\Users\\anves\\Pictures\\Lab2\\Data.txt") as f2,  open("C:\\Users\\anves\\Downloads\\NLP\\Tutorial-2-CoreNLP\\the-file-name.txt") as f1:
        k = 0
        for line in f2:
            list.append(line)
        for line1 in f1:
            i = 0
            y_true = list[i]
            i= i+1
            y_pred = (line1)
            BLEUscore = sentence_bleu(word_tokenize(y_true), word_tokenize(y_pred), weights=(1, 0, 0, 0))
            [precision, recall, f_score] = r.rouge_l([y_true], [y_pred])
            print("Precision is :" + str(precision) + "\nRecall is :" + str(recall) + "\nF Score is :" + str(f_score))
            print(BLEUscore)


if __name__ == "__main__":
    accuracy()




# for i in range(3):
            #     y_true_line.append(list[k])
            #     y_true.append(word_tokenize(list[k]))
            #     k = k + 1
            # print(y_true)