import unittest
import app as app

class TestaApp(unittest.TestCase):
    def remove_stopwords_from_data(self):
        """
        Testa a 
        :return:
        """

        app = App()

        app.set_stopwords(nltk.corpus.stopwords.words('portuguese'))

        phrase = 'por mais que eu já tenha mil reais, gostaria de mais'

        for word in phrase:
            if(word in app.get_stopwords()):
                self.assertNotIn(word, app.remove_stopwords_from_data(phrase))

if __name__ == '__main__':
    unittest.main()