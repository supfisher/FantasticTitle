import nltk
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import words
import re
from itertools import combinations, permutations
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--title", default='Generate Fantastic Title by Using a Single Line Command', type=str,
                        help="This is your paper title")
parser.add_argument("--word_len", default=5, type=int,
                        help="This is the generated word length")
parser.add_argument("--mnc", "--max_num_char", default=1, type=int,
                        help="This is the max number of character allowed to select in one word.")
parser.add_argument("--num_threads", default=10, type=int,
                        help="This is the number of threads to run the code")
parser.add_argument("--constraint", default='mcow', type=str,
                        help="This is the type of the constraint to generate an abbreviate word")

args = parser.parse_args()

dictionary = set(words.words())


class CheckWord (threading.Thread):
    def __init__(self, words):
        threading.Thread.__init__(self)
        self.words = words
        self.valid_words = []

    def run(self):
        for word in self.words:
            word = ''.join(word).lower()
            if self.is_english_word(word) and word not in self.valid_words:
                self.valid_words.append(word)

    def is_english_word(self, word):
        return word in dictionary


class BeautifulTitle:
    def __init__(self, sentence, word_len=3, constraint='ocow', num_threads=10, max_num_char=1):
        self.word_len = word_len
        self.constraint = constraint
        self.max_num_char = max_num_char
        sents = self.preprocess_sentence(sentence)
        words = self.generate_words(sents)
        self.threads = []
        for _ in range(num_threads):
            self.threads.append(CheckWord(words[:min(len(words), int(len(words)/num_threads)+1)]))
            del words[:min(len(words), int(len(words)/num_threads)+1)]

        self.valid_words = []

    def preprocess_sentence(self, sentence):
        from nltk.corpus import stopwords
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()

        from nltk import word_tokenize, pos_tag
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return None

        sents = re.split('[^a-zA-Z]', sentence)
        tagged_sent = pos_tag(sents)  # 获取单词词性
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            if tag[0].lower() not in stopwords.words('english'):
                lemmas_sent.append(wnl.lemmatize(tag[0].lower(), pos=wordnet_pos))  # 词形还原
        return lemmas_sent

    def generate_words(self, sentence):
        if self.constraint == 'random':
            return self.random_generate_words(sentence)
        elif self.constraint == 'mcow':
            return self.mcow_generate_words(sentence, max_num_char=self.max_num_char)

    def random_generate_words(self, sents):
        sentence = ''.join(re.split(r'[^A-Za-z]', sents))
        return list(combinations(sentence, self.word_len))

    def mcow_generate_words(self, sents, max_num_char):
        """
            multiple character from one word
        """

        def itera_sent(sents):
            if len(sents) > 1:
                for c in self.__iter_word(sents[0], max_num_char):
                    tmp = c
                    for candidate in itera_sent(sents[1:]):
                        yield (tmp + candidate).lower()
            else:
                for c in self.__iter_word(sents[0], max_num_char):
                    yield c

        candidates = []
        for candidate in itera_sent(sents):
            candidates.extend(self.random_generate_words(candidate))

        return candidates

    def __iter_word(self, word, max_num_char=1):
        for num_char in range(1, max_num_char+1):
            for chars in combinations(word, num_char):
                yield ''.join(chars)

    def __call__(self, *args, **kwargs):
        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()

        for thread in self.threads:
            self.valid_words.extend(thread.valid_words)

        return sorted(set(self.valid_words))


if __name__ == '__main__':
    bt = BeautifulTitle(args.title, word_len=args.word_len, constraint=args.constraint, num_threads=args.num_threads)
    valid_words = bt()
    print(valid_words)