import re
def build_corpus(filename):
    words = list()
    with open(filename, 'r') as raw_file:
        article = raw_file.read()
        article = re.sub('!', '.', article)
        article = re.sub('\?', '.', article)
        article = re.sub(',', '.', article)
        article = re.sub('"', '', article)
        article = article.lower()
        article = article.split('.')
        for line_idx, line in enumerate(article):
            article[line_idx] = line.split()
    return article
