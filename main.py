import gc
import os
import pickle
import re
import time
from operator import concat

import pytesseract
from PIL import Image
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, classification_report, roc_curve, auc
from flask import Flask, request, jsonify, send_from_directory

UPLOAD_DIRECTORY = "data/api_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

api = Flask(__name__)


class DocClassifier:
    def __init__(self):
        self._mystem = Mystem(entire_input=True)
        self._stopwords = self._read_stopwords()
        self._vect_word = None
        self._vect_char = None
        self._model = None

    def get_model(self):
        return self._model

    def train(self):
        with open('data/docs/classes_lemmatized.pickle', 'rb') as file:
            classes_lemmatized = pickle.load(file)

        # removes files where an error occurred
        for cls in classes_lemmatized:
            classes_lemmatized[cls] = [x for x in classes_lemmatized[cls] if x is not None]

        data = []
        for cls in (x for x in classes_lemmatized if x != 'order'):
            for text in classes_lemmatized[cls]:
                data.append([' '.join(text), cls])

        data = pd.DataFrame(data, columns=['text', 'label'])
        gc.collect()

        # fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        # ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
        # sns.countplot(data['label'], palette='magma', ax=ax1)

        # train, test = train_test_split(data, test_size=0.2)
        train = data

        self._vect_word = TfidfVectorizer(max_features=20, lowercase=True, analyzer='word',
                                          ngram_range=(1, 3), dtype=np.float32)
        self._vect_char = TfidfVectorizer(max_features=40, lowercase=True, analyzer='char',
                                          ngram_range=(3, 6), dtype=np.float32)

        # Word ngram vector
        tr_vect_word = self._vect_word.fit_transform(train['text'])
        # ts_vect_word = self._vect_word.transform(test['text'])

        # Character n gram vector
        tr_vect_char = self._vect_char.fit_transform(train['text'])
        # ts_vect_char = self._vect_char.transform(test['text'])
        gc.collect()

        x_train = sparse.hstack([tr_vect_word, tr_vect_char])
        # x_test = sparse.hstack([ts_vect_word, ts_vect_char])

        target_col = 'label'
        y_train = train[target_col]
        # y_test = test[target_col]

        del tr_vect_word, tr_vect_char
        # del ts_vect_word, ts_vect_char
        gc.collect()

        self._model = LogisticRegression(C=2, random_state=0, class_weight='balanced', multi_class='multinomial')
        self._model.fit(x_train, y_train)
        print(self._model.score)
        # cv_score.append(lr.score)
        # prd[:, i] = lr.predict_proba(x_test)[:, 1]

        pred = self._model.predict(x_train)
        print('\nConfusion matrix\n', confusion_matrix(y_train, pred))
        print(classification_report(y_train, pred))

        # pred = self._model.predict(x_test)
        # print('\nConfusion matrix\n', confusion_matrix(y_test, pred))
        # print(classification_report(y_test, pred))

    def predict(self, filename):
        text = self.lemmatize_image(Image.open(os.path.join('data/api_uploaded_files/', filename)))
        data = pd.DataFrame([text], columns=['text  '])
        tr_vect_word = self._vect_word.transform(data['text'])
        tr_vect_char = self._vect_char.transform(data['text'])
        x = sparse.hstack([tr_vect_word, tr_vect_char])
        pred = self._model.predict(x)

        return pred[0]

    def prepare_lemmatized(self):
        acts = self.read_images('data/docs/acts')
        invoice = self.read_images('data/docs/invoice')
        order = self.read_images('data/docs/order')
        receipt = self.read_images('data/docs/receipt')
        reference = self.read_images('data/docs/reference')

        classes = {
            'acts': acts,
            'invoice': invoice,
            'order': order,
            'receipt': receipt,
            'reference': reference
        }

        classes_lemmatized = {}
        total = sum((len(classes[x]) for x in classes))
        current = 1
        for key in classes:
            lemmatized_texts = []
            for image in classes[key]:
                lemmatized_texts.append(self.lemmatize_image(image))
                print('%d/%d' % (current, total))
                current += 1
            classes_lemmatized[key] = lemmatized_texts

        with open('data/docs/classes_lemmatized.pickle', 'wb') as file:
            pickle.dump(classes_lemmatized, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _read_stopwords():
        with open('data/stopwords/stop-words-russian.txt', "r") as file:
            lines = file.readlines()
            return set((x.strip() for x in lines))

    @staticmethod
    def read_images(directory):
        result = []
        for filename in os.listdir(directory):
            result.append(Image.open(os.path.join(directory, filename)))
        return result

    def lemmatize_image(self, image):
        try:
            text = pytesseract.image_to_string(image, lang='rus')
            text = re.sub(r'(\W+|_)',  # ["!#$%&\'()*+,-./:;<=>?@\\\\[\\]^_`{|}~№\n\r\t« »‚‘’„——°]+
                          ' ',
                          text,
                          flags=re.RegexFlag.MULTILINE | re.RegexFlag.UNICODE)
            lemmas = (x for x in self._mystem.lemmatize(text) if not x.isspace())
            lemmas = [x for x in lemmas if x not in self._stopwords]
            return lemmas
        except:
            return None


@api.route("/files", methods=["POST"])
def post_file():
    """Upload a file."""

    file = request.files.get('file')
    if file:
        classifier = DocClassifier
        filename = str(time.time()) + file.filename
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        file.save(path)
        result = classifier.predict(filename)
    # Return 201 CREATED
    return result, 201


def main():
    classifier = DocClassifier()
    classifier.train()
    print(classifier.predict('1.jpg'))
    api.run(debug=True, port=5000)


if __name__ == '__main__':
    main()
