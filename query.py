from boolparser import parse_boolean
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import copy
import argparse
import ast
from collections import defaultdict
import math
import pytest
import os


def getstopwords(file):
    stopwords = []
    with open(file,'r') as f:
        for line in f:
            word = line.strip()
            stopwords.append(word)
    return stopwords


def readindexfile(file, external=True):
    if external:
        with open(file) as index_file:
            inverted_index = ast.literal_eval(index_file.readline())
            return inverted_index[0], inverted_index[1]
    else:
        inverted_index = ast.literal_eval(file)
        return inverted_index[0], inverted_index[1]


def readtitle(file):
    mytitle = {}
    file = open(file,'r')
    for line in file:
        line = line.strip()
        docid,title = line.split('|')
        mytitle[docid] = title
    return mytitle


def text_process(text,sw, verbose=True):
    ps = PorterStemmer()
    text = text.replace(" OR ", " ")
    text = text.replace(" AND ", " ")
    line = text.lower()
    line = re.sub(r'[^a-z0-9 ]', ' ', line)
    split = word_tokenize(line)
    # stream = [i for i in split if i not in sw]
    stream = []
    if verbose:
        for word in split:
            if word not in sw:
                stream.append(word)
            else:
                if word == '':
                    continue
                else:
                    print('FYI: \'{}\' is a stop word and is being removed from your query'.format(word))
    else:
        stream = [i for i in split if i not in sw]
    query = [ps.stem(word) for word in stream]
    return query


def mergetwo(a, b):
    return list(set(a) & set(b))


def intersection(lists):
    results = mergetwo(lists[0], lists[1])
    i = 1
    if i < (len(lists)-1):
        results = mergetwo(results, lists[i+1])
    else:
        results = results
    return results


def getdocid(query, inverted_index):
    postings = inverted_index.get(query, [])
    doc = [item[0] for item in postings]
    return doc


def dotProduct(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0
    return sum([a * b for a, b in zip(vec1, vec2)])


def rankdoc(queryterm, docs, total_docs, inverted_index):
    """
    :param: queryterm : processed_query [q1,q2,q3]
    :param: docs: matched docid list doc = [id1,id2...] result from query without ranking
    """

    queryVec = [0] * len(queryterm)
    docVec = defaultdict(lambda: [0] * len(queryterm))
    # for example x  = defaultdict(lambda :[0]*3)
    # can have x looks like {docid1: [tfidf1, tfidf2, tfidf3], docid2: [tfidf1, tfidf2, tfidf3]})

    for position, term in enumerate(queryterm):
        posting_dict = inverted_index.get(term, ast.literal_eval("[0, {'dummy': [0, [0], 0]}]"))
        # print(posting_dict)
        # for example: [2, {1: [1, [3]], 12: [2, [2204, 2219]]}]
        # calculate idf for each term in query:
        if posting_dict[0] == 0:
            queryVec[position] = 0
        else:
            queryVec[position] = math.log(total_docs / posting_dict[0])

        for docid in docs:
            if docid in posting_dict[1]:
                docVec[docid][position] = posting_dict[1][docid][0]
            else:
                docVec[docid][position] = 0

    # Cosine Similarity(Query,Document1) = Dot product(Query, Document1) / ||Query|| * ||Document1||
    scores = [[docid, dotProduct(one_docVec, queryVec) / (
    math.sqrt(dotProduct(one_docVec, one_docVec)) * math.sqrt(dotProduct(queryVec, queryVec)))] for docid, one_docVec in
              docVec.items()]
    id_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return id_scores


def pq(doclist):
    doclist = copy.deepcopy(doclist)
    for item in doclist[0]:
        item[1] = [x+1 for x in item[1]]   # increment position by 1
    result=[]
    for i in range(len(doclist[0])):
        li = intersection([x[i][1] for x in doclist] )  # find intersection
        if li==[]:
            continue
        else:
            result.append([doclist[0][i][0], li])
    return result


def OneWordQuery(query, sw, inverted_index_new,total_docs):

    q = text_process(query, sw)

    if len(q) == 0:
        return None
    elif q[0] in inverted_index_new:
        Df = inverted_index_new[q[0]][0]
        docposting = inverted_index_new[q[0]][1]

        match_list = []
        for key, value in docposting.items():
            Tf_normal = float(value[0]/value[2])
            IDF = math.log(total_docs/Df)
            match_list.append([key, Tf_normal*IDF])
        return match_list
    else:
        return None


def FreeTextQuery(query, sw, inverted_index,inverted_index_new,total_docs):
    doc = set()
    q = text_process(query,sw)
    for i in q:
        if i in inverted_index:
            docposting = inverted_index[i]
            match = [item[0] for item in docposting]
            doc.update(match)
    doc = sorted(doc)
    res = rankdoc(q, doc, total_docs, inverted_index_new)
    return res


def PhraseQuery(query, sw, inverted_index,inverted_index_new,total_docs):
    query = text_process(query, sw)
    if len(query) == 1:
        print('Query consisted of too many stop or similar words .  Only one search term remains: {}\n'
              'Reclassifying as a single word query.'.format(query[0]))
        return OneWordQuery(query[0], sw, inverted_index_new, total_docs)
    if len(query) == 0:
        return []

    doc = []
    for i in range(len(query)):  # get docid for each word in query
        docposting = getdocid(query[i], inverted_index)
        doc.append(docposting)
    docid = intersection(doc)  # docid that all terms appear
    postings = [inverted_index.get(term, []) for term in query]
    doc_position = []
    for i in range(len(postings)):
        doc_position.append([x for x in postings[i] if x[0] in docid])
    # doc_position contains intersection docid and corresponding position

    results = pq([doc_position[0], doc_position[1]])
    i = 1
    if i < (len(doc_position) - 1):
        results = pq([results, doc_position[i + 1]])
    else:
        results = results
    matched = [x[0] for x in results]
    res = rankdoc(query, matched, total_docs, inverted_index_new)
    return res


def intersect(a, b):  # for AND operator
    return list(set(a) & set(b))


def union(a, b):  # for OR operator
    return list(set(a) | set(b))


def intersect_many(list_of_lists):  # for AND operator
    return list(set.intersection(*[set(doc_list) for doc_list in list_of_lists]))


def union_many(list_of_lists):  # for OR operator
    return list(set.union(*[set(doc_list) for doc_list in list_of_lists]))


def Boolquery(query,sw,inverted_index):

    if isinstance(query, tuple):
        if query[0] == 'AND':
            doc_list = [Boolquery(word, sw, inverted_index) for word in query[1]]
            # print(doc_list)
            return intersect_many(doc_list)
        else:
            doc_list = [Boolquery(word, sw, inverted_index) for word in query[1]]
            return union_many(doc_list)
    if isinstance(query, str):
        query = text_process(query, sw)
        if len(query) != 0:
            doc = getdocid(query[0], inverted_index)
        else:
            doc = []
        return doc

class QueryFactory:
    def create(query_string: str):
        if not query_string:
            return None
        if query_string[0] == '"' and query_string[-1] == '"':
            if len(query_string[1:-2].split()) <= 1:
                return 'OWQ'
            else:
                return 'PQ'
        if any([t in query_string for t in ('(', ')', ' AND ', ' OR ')]):
            return 'BQ'
        words = query_string.split()
        if len(words) == 1:
            return 'OWQ'
        elif len(words) > 1:
            return 'FTQ'
        else:
            raise ValueError('This query string is not supported.')

def rank_by_pr_score(file,result):
    """
    :param: file : scores.dat
    :param: result: query result in format: [[doc1,tfidf score],[doc2,tfidf score],[doc3, tfidf score]]
    only need matched docid from result

    """
    #read in pagerank scores
    prscore = {}
    file = open(file, 'r')
    for line in file:
        line = line.strip()
        docid, score = line.split('|')
        prscore[int(docid)] = float(score)
    # extract docid from result first
    matcheddoc = [x[0]for x in result]
    # Sort the documents by decreasing PageRank score. notice: some documents don't have pagerank score
    doc_with_score = list(prscore.keys())
    doc_intersection = list(set(doc_with_score) & set(matcheddoc))
    doc_zero_rank = list(set(matcheddoc) - set(doc_intersection))
    doc_pr = [[k, prscore[k]] for k in doc_intersection]
    # sort by decreasing pagerank score
    doc_pr.sort(key=lambda x: x[1], reverse=True)
    newresult = [x[0] for x in doc_pr] + doc_zero_rank
    return newresult

################## test units start here  ##################################

def test_stop_word_removal_and_lemm():
    stop_words = ['one', 'two', '123']
    assert text_process('one', stop_words) == []
    assert text_process('tWo', stop_words) == []
    assert text_process('tWo one', stop_words) == []
    assert text_process('tWo', stop_words) == []
    assert text_process('123 dog cat', stop_words) == ['dog', 'cat']
    assert text_process('dory', stop_words) == ['dori']
    assert text_process('dory908049834023dog', stop_words) == ['dory908049834023dog']
    assert text_process('dory908049834023dog %#$# fsd ((', stop_words) == ['dory908049834023dog', 'fsd']

def test_query_classing():
    assert QueryFactory.create('DOG') == 'OWQ'
    assert QueryFactory.create('the dog walks fast') == 'FTQ'
    assert QueryFactory.create('(') == 'BQ'
    assert QueryFactory.create('(asd') == 'BQ'
    assert QueryFactory.create('dog OR cat') == 'BQ'
    assert QueryFactory.create('dog ORange') == 'FTQ'
    assert QueryFactory.create('ANDerson Cooper') == 'FTQ'
    assert QueryFactory.create('\"the greatest query\"') == 'PQ'
    with pytest.raises(ValueError):
        QueryFactory.create('       ')

def test_query_results():
    stop_words = ['one', 'two', '123']
    total_docs, inverted_index_new_test = readindexfile("(13,{'2001': [2, {0: [1, [0], 3], 3: [1, [6], 12]}], 'space': [4, {0: [1, [1], 3], 2: [1, [5], 6], 3: [1, [7], 12], 8: [1, [3], 4]}], 'odyssey': [2, {0: [1, [2], 3], 3: [1, [8], 12]}], 'armstrong': [1, {1: [1, [0], 7]}], 'first': [2, {1: [2, [1, 5], 7], 2: [1, [2], 6]}], 'man': [2, {1: [3, [2, 4, 6], 7], 2: [1, [3], 6]}], 'moon': [1, {1: [1, [3], 7]}], 'yuri': [1, {2: [1, [0], 6]}], 'gagarin': [1, {2: [1, [1], 6]}], 'orbit': [1, {2: [1, [4], 6]}], 'kubrik': [1, {3: [1, [0], 12]}], 'movi': [1, {3: [1, [1], 12]}], 'includ': [1, {3: [1, [2], 12]}], 'full': [2, {3: [2, [3, 9], 12], 7: [1, [0], 6]}], 'metal': [2, {3: [2, [4, 10], 12], 7: [1, [1], 6]}], 'jacket': [2, {3: [1, [5], 12], 7: [1, [2], 6]}], 'jump': [1, {3: [1, [11], 12]}], 'danger': [1, {4: [1, [0], 7]}], 'killer': [3, {4: [1, [1], 7], 5: [1, [0], 5], 7: [1, [4], 6]}], 'kiss': [2, {4: [5, [2, 3, 4, 5, 6], 7], 5: [1, [3], 5]}], 'murder': [1, {5: [1, [1], 5]}], 'woman': [1, {5: [1, [2], 5]}], 'cat': [1, {5: [1, [4], 5]}], 'dog': [1, {6: [1, [0], 4]}], 'kill': [1, {6: [1, [1], 4]}], 'child': [1, {6: [1, [2], 4]}], 'snorf': [1, {6: [1, [3], 4]}], 'bullet': [1, {7: [1, [3], 6]}], 'hippi': [1, {7: [1, [5], 6]}], 'recent': [1, {8: [1, [0], 4]}], 'nasa': [1, {8: [1, [1], 4]}], 'program': [1, {8: [1, [2], 4]}], 'clockwork': [4, {9: [1, [0], 2], 10: [1, [2], 5], 11: [1, [3], 66], 12: [1, [3], 66]}], 'orang': [2, {9: [1, [1], 2], 10: [1, [0], 5]}], 'swiss': [1, {10: [1, [1], 5]}], 'loud': [1, {10: [1, [3], 5]}], 'nois': [1, {10: [1, [4], 5]}], 'fear': [2, {11: [1, [0], 66], 12: [1, [0], 66]}], 'destruct': [2, {11: [1, [1], 66], 12: [1, [1], 66]}], 'glori': [2, {11: [1, [2], 66], 12: [1, [2], 66]}], 'mater': [2, {11: [6, [4, 20, 29, 37, 46, 58], 66], 12: [6, [4, 20, 29, 37, 46, 58], 66]}], 'time': [2, {11: [5, [5, 7, 35, 45, 56], 66], 12: [5, [5, 7, 35, 45, 56], 66]}], 'travel': [2, {11: [3, [6, 36, 57], 66], 12: [3, [6, 36, 57], 66]}], 'tell': [2, {11: [1, [8], 66], 12: [1, [8], 66]}], 'stori': [2, {11: [4, [9, 22, 26, 48], 66], 12: [4, [9, 22, 26, 48], 66]}], 'lightn': [2, {11: [1, [10], 66], 12: [1, [10], 66]}], 'mcqueen': [2, {11: [7, [11, 13, 23, 31, 39, 49, 63], 66], 12: [7, [11, 13, 23, 31, 39, 49, 63], 66]}], 'take': [2, {11: [2, [12, 38], 66], 12: [2, [12, 38], 66]}], 'help': [2, {11: [1, [14], 66], 12: [1, [14], 66]}], 'him': [2, {11: [2, [15, 52], 66], 12: [2, [15, 52], 66]}], 'save': [2, {11: [1, [16], 66], 12: [1, [16], 66]}], 'radiat': [2, {11: [1, [17], 66], 12: [1, [17], 66]}], 'spring': [2, {11: [1, [18], 66], 12: [1, [18], 66]}], 'howev': [2, {11: [2, [19, 61], 66], 12: [2, [19, 61], 66]}], 'finish': [2, {11: [2, [21, 47], 66], 12: [2, [21, 47], 66]}], 'there': [2, {11: [1, [24], 66], 12: [1, [24], 66]}], 'listen': [2, {11: [1, [25], 66], 12: [1, [25], 66]}], 'never': [2, {11: [1, [27], 66], 12: [1, [27], 66]}], 'shown': [2, {11: [1, [28], 66], 12: [1, [28], 66]}], 'brought': [2, {11: [1, [30], 66], 12: [1, [30], 66]}], 'back': [2, {11: [3, [32, 40, 43], 66], 12: [3, [32, 40, 43], 66]}], 'could': [2, {11: [2, [33, 62], 66], 12: [2, [33, 62], 66]}], 'skip': [2, {11: [1, [34], 66], 12: [1, [34], 66]}], 'present': [2, {11: [1, [41], 66], 12: [1, [41], 66]}], 'go': [2, {11: [1, [42], 66], 12: [1, [42], 66]}], 'own': [2, {11: [1, [44], 66], 12: [1, [44], 66]}], 'doesn': [2, {11: [1, [50], 66], 12: [1, [50], 66]}], 'believ': [2, {11: [1, [51], 66], 12: [1, [51], 66]}], 'even': [2, {11: [1, [53], 66], 12: [1, [53], 66]}], 'though': [2, {11: [1, [54], 66], 12: [1, [54], 66]}], 'went': [2, {11: [1, [55], 66], 12: [1, [55], 66]}], 'just': [2, {11: [1, [59], 66], 12: [1, [59], 66]}], 'now': [2, {11: [1, [60], 66], 12: [1, [60], 66]}], 'think': [2, {11: [1, [64], 66], 12: [1, [64], 66]}], 'head': [2, {11: [1, [65], 66], 12: [1, [65], 66]}]})",False)
    inverted_index_test = {}
    for term in list(inverted_index_new_test.keys())[1:]:
        inverted_index_test[term] = [[docid, position[1]] for docid, position in inverted_index_new_test[term][1].items()]
    assert OneWordQuery('orange', stop_words, inverted_index_new_test, total_docs) == [[9, 0.9359010884507957], [10, 0.37436043538031827]]
    assert FreeTextQuery('orange killer', stop_words, inverted_index_test,inverted_index_new_test, total_docs) == [[9, 0.7872092639569277], [10, 0.7872092639569277], [4, 0.6166859611993708], [5, 0.6166859611993708], [7, 0.6166859611993708]]
    assert PhraseQuery('"metal jacket"',stop_words, inverted_index_test,inverted_index_new_test,total_docs)== [[7, 1.0], [3, 0.9486832980505139]]
    assert Boolquery('orange And man',stop_words, inverted_index_test) == [9, 10]
    assert Boolquery('orange man', stop_words, inverted_index_test) == [9, 10]
    assert Boolquery('and', stop_words, inverted_index_test) == []
    assert OneWordQuery('and', stop_words, inverted_index_test, total_docs) == None
    assert FreeTextQuery('and', stop_words, inverted_index_test,inverted_index_new_test, total_docs) == []
    assert PhraseQuery('and orange', stop_words, inverted_index_test, inverted_index_new_test, total_docs) == []
    assert PhraseQuery('and 12398712391287sSSSEW888888777 orange', stop_words, inverted_index_test, inverted_index_new_test, total_docs) == []

def test_sets():
    assert union([1, 2, 3], [3, 4, 5]) == [1, 2, 3, 4, 5]
    assert sorted(union([8, 2, 8], [3, 4, 5])) == [2, 3, 4, 5, 8]
    assert intersect([8, 2, 8], [3, 4, 5]) == []
    assert intersect([8, 2, 8], [2, 4, 5]) == [2]
    assert sorted(union([], [3, 4, 5])) == [3, 4, 5]
    assert intersect([], [3, 4, 5]) == []
    assert intersect_many([[], [3, 4, 5]]) == []
    assert sorted(union_many([[], [3, 4, 5]])) == [3, 4, 5]
    assert intersect_many([[], [3, 4, 5], [4, 5, 6]]) == []
    assert sorted(union_many([[], [3, 4, 5], [22, 33, 66], [44, 22, 66]])) == [3, 4, 5, 22, 33, 44, 66]

def test_dotProduct():
    assert dotProduct([1,2,3],[3,4,5])== 26
    assert dotProduct([0.2,1.3,3.4],[2.4,3.2,4.9]) == 21.3

def test_rankdoc():
    total_docs, inverted_index_new_test = readindexfile("(13,{'2001': [2, {0: [1, [0], 3], 3: [1, [6], 12]}], 'space': [4, {0: [1, [1], 3], 2: [1, [5], 6], 3: [1, [7], 12], 8: [1, [3], 4]}], 'odyssey': [2, {0: [1, [2], 3], 3: [1, [8], 12]}], 'armstrong': [1, {1: [1, [0], 7]}], 'first': [2, {1: [2, [1, 5], 7], 2: [1, [2], 6]}], 'man': [2, {1: [3, [2, 4, 6], 7], 2: [1, [3], 6]}], 'moon': [1, {1: [1, [3], 7]}], 'yuri': [1, {2: [1, [0], 6]}], 'gagarin': [1, {2: [1, [1], 6]}], 'orbit': [1, {2: [1, [4], 6]}], 'kubrik': [1, {3: [1, [0], 12]}], 'movi': [1, {3: [1, [1], 12]}], 'includ': [1, {3: [1, [2], 12]}], 'full': [2, {3: [2, [3, 9], 12], 7: [1, [0], 6]}], 'metal': [2, {3: [2, [4, 10], 12], 7: [1, [1], 6]}], 'jacket': [2, {3: [1, [5], 12], 7: [1, [2], 6]}], 'jump': [1, {3: [1, [11], 12]}], 'danger': [1, {4: [1, [0], 7]}], 'killer': [3, {4: [1, [1], 7], 5: [1, [0], 5], 7: [1, [4], 6]}], 'kiss': [2, {4: [5, [2, 3, 4, 5, 6], 7], 5: [1, [3], 5]}], 'murder': [1, {5: [1, [1], 5]}], 'woman': [1, {5: [1, [2], 5]}], 'cat': [1, {5: [1, [4], 5]}], 'dog': [1, {6: [1, [0], 4]}], 'kill': [1, {6: [1, [1], 4]}], 'child': [1, {6: [1, [2], 4]}], 'snorf': [1, {6: [1, [3], 4]}], 'bullet': [1, {7: [1, [3], 6]}], 'hippi': [1, {7: [1, [5], 6]}], 'recent': [1, {8: [1, [0], 4]}], 'nasa': [1, {8: [1, [1], 4]}], 'program': [1, {8: [1, [2], 4]}], 'clockwork': [4, {9: [1, [0], 2], 10: [1, [2], 5], 11: [1, [3], 66], 12: [1, [3], 66]}], 'orang': [2, {9: [1, [1], 2], 10: [1, [0], 5]}], 'swiss': [1, {10: [1, [1], 5]}], 'loud': [1, {10: [1, [3], 5]}], 'nois': [1, {10: [1, [4], 5]}], 'fear': [2, {11: [1, [0], 66], 12: [1, [0], 66]}], 'destruct': [2, {11: [1, [1], 66], 12: [1, [1], 66]}], 'glori': [2, {11: [1, [2], 66], 12: [1, [2], 66]}], 'mater': [2, {11: [6, [4, 20, 29, 37, 46, 58], 66], 12: [6, [4, 20, 29, 37, 46, 58], 66]}], 'time': [2, {11: [5, [5, 7, 35, 45, 56], 66], 12: [5, [5, 7, 35, 45, 56], 66]}], 'travel': [2, {11: [3, [6, 36, 57], 66], 12: [3, [6, 36, 57], 66]}], 'tell': [2, {11: [1, [8], 66], 12: [1, [8], 66]}], 'stori': [2, {11: [4, [9, 22, 26, 48], 66], 12: [4, [9, 22, 26, 48], 66]}], 'lightn': [2, {11: [1, [10], 66], 12: [1, [10], 66]}], 'mcqueen': [2, {11: [7, [11, 13, 23, 31, 39, 49, 63], 66], 12: [7, [11, 13, 23, 31, 39, 49, 63], 66]}], 'take': [2, {11: [2, [12, 38], 66], 12: [2, [12, 38], 66]}], 'help': [2, {11: [1, [14], 66], 12: [1, [14], 66]}], 'him': [2, {11: [2, [15, 52], 66], 12: [2, [15, 52], 66]}], 'save': [2, {11: [1, [16], 66], 12: [1, [16], 66]}], 'radiat': [2, {11: [1, [17], 66], 12: [1, [17], 66]}], 'spring': [2, {11: [1, [18], 66], 12: [1, [18], 66]}], 'howev': [2, {11: [2, [19, 61], 66], 12: [2, [19, 61], 66]}], 'finish': [2, {11: [2, [21, 47], 66], 12: [2, [21, 47], 66]}], 'there': [2, {11: [1, [24], 66], 12: [1, [24], 66]}], 'listen': [2, {11: [1, [25], 66], 12: [1, [25], 66]}], 'never': [2, {11: [1, [27], 66], 12: [1, [27], 66]}], 'shown': [2, {11: [1, [28], 66], 12: [1, [28], 66]}], 'brought': [2, {11: [1, [30], 66], 12: [1, [30], 66]}], 'back': [2, {11: [3, [32, 40, 43], 66], 12: [3, [32, 40, 43], 66]}], 'could': [2, {11: [2, [33, 62], 66], 12: [2, [33, 62], 66]}], 'skip': [2, {11: [1, [34], 66], 12: [1, [34], 66]}], 'present': [2, {11: [1, [41], 66], 12: [1, [41], 66]}], 'go': [2, {11: [1, [42], 66], 12: [1, [42], 66]}], 'own': [2, {11: [1, [44], 66], 12: [1, [44], 66]}], 'doesn': [2, {11: [1, [50], 66], 12: [1, [50], 66]}], 'believ': [2, {11: [1, [51], 66], 12: [1, [51], 66]}], 'even': [2, {11: [1, [53], 66], 12: [1, [53], 66]}], 'though': [2, {11: [1, [54], 66], 12: [1, [54], 66]}], 'went': [2, {11: [1, [55], 66], 12: [1, [55], 66]}], 'just': [2, {11: [1, [59], 66], 12: [1, [59], 66]}], 'now': [2, {11: [1, [60], 66], 12: [1, [60], 66]}], 'think': [2, {11: [1, [64], 66], 12: [1, [64], 66]}], 'head': [2, {11: [1, [65], 66], 12: [1, [65], 66]}]})", False)
    assert rankdoc(['2001', 'space'],[0,3],total_docs,inverted_index_new_test ) == [[0, 0.9751424258351847],[3, 0.9751424258351847]]

def test_rank_by_pr_score():
    file = 'index/scores.dat'
    # result = [[doc1,tfidf],[doc2,tfidf]]
    result = [[1461, 0.007344903412270338], [1819, 0.004388785468822794], [1829, 0.0014737439142389383],
              [1831, 0.0015144029826445705], [1832, 0.0019216244848270273]]
    result2 = [[1461, 0.022961158908791718], [1833, 0.08012529853146634], [1926, 0.03595234092297651],
              [1945, 0.03288973410361184], [1981, 0.03669515788419503], [2034, 0.030834125722136098],
              [2132, 0.22200570519937993], [2133, 0.11037571783923975], [2135, 0.35053532399902093],
              [2138, 0.0925023771664083], [2141, 0.1305915912937529], [2158, 0.31077226632897914],
              [2159, 0.11566143830937305], [2161, 0.0474878513795465], [2162, 0.053495350650452994],
              [2163, 0.053819564896819376], [2164, 0.08072934734522906]]
    assert rank_by_pr_score(file, result) == [1829, 1831, 1832, 1819, 1461]
    assert rank_by_pr_score(file,result2) == [1833, 1945, 1926, 2159, 2034, 2158, 2133, 1981, 1461, 2161, 2163, 2138, 2162, 2164, 2135, 2132, 2141]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("stop_word_collection", nargs='?', default=None)
    parser.add_argument("index_file", nargs='?', default=None)

    parser.add_argument("-t", help="output titles instead of ids", action="store_true")
    parser.add_argument("-v", help="output title/score pairs, with each pair appearing on a new line", action="store_true")
    parser.add_argument("-z", help="output top 20 title/score pairs, with each pair appearing on a new line", action="store_true")
    parser.add_argument("--rank", nargs='?', default='tfidf',
                        help="output docid sorted by decreasing pagerank score")
    args = parser.parse_args()

    if args.stop_word_collection is None or args.index_file is None:
        print('2 argument files are needed\n'
              '1. stop_word_collection\n'
              '2. index file\n')
    else:
        stopwords = args.stop_word_collection
        filepath = args.index_file

        # get files inside folder
        myindex = str(filepath)+'/'+ 'inverted_index.dat'
        mytitle = str(filepath)+'/'+ 'title_index.dat'
        pr_scores = str(filepath)+'/'+ 'scores.dat'

    rank_pagerank = False

    if args.rank.lower() != 'tfidf' and args.rank.lower() != 'pagerank':
        print("Invalid '--rank' argument.  Please choose 'pagerank' or 'tfidf'\n"
              "Alternatively, leave out the argument to run TFIDF by default.")
        exit()
    else:
        if args.rank.lower() == 'pagerank':
            rank_pagerank = True

    #read stopwords
    sw = getstopwords(stopwords)
    #read inverted index
    total_docs, inverted_index_new = readindexfile(myindex)
    #read title_id
    title_index = readtitle(mytitle)

    inverted_index = {}
    for term in list(inverted_index_new.keys())[1:]:
        inverted_index[term] = [[docid, position[1]] for docid, position in inverted_index_new[term][1].items()]

    while True:
        try:
            query = input()
        except EOFError:  # catch the ctrl-D exit
            exit()

        q = QueryFactory.create(query)
        result = []
        if q == 'OWQ':
            result = OneWordQuery(query, sw, inverted_index_new, total_docs)

        if q == 'FTQ':
            result = FreeTextQuery(query, sw, inverted_index,inverted_index_new, total_docs)

        if q == 'PQ':
            result = PhraseQuery(query, sw, inverted_index,inverted_index_new,total_docs)

        if q =='BQ':
            try:
                q = parse_boolean(query)
            except Exception as ex:
                print('ERROR: Boolean query ‘{}’ could not be parsed.\nPlease try again.'.format(ex.args[0]))
                continue
            doc = Boolquery(q, sw, inverted_index)
            query = text_process(query, sw, verbose=False)
            res = rankdoc(query, doc, total_docs, inverted_index_new)
            result = res

            # catch queries that return zero results
        if result is None or len(result) == 0:
            print('Query returned no results. Please try again.')
            continue
            # if there's more than one result, sort base on score.
        elif len(result) > 1:
            result.sort(key=lambda x: x[1], reverse=True)

        if args.rank.lower() == 'tfidf':
            if not args.t and not args.v:
                print(' '.join([str(x[0]) for x in result]))
            if args.v:
                for item in result:
                    replace = title_index[str(item[0])]
                    item[0] = replace
                    print(': '.join([str(x) for x in item]))
            if args.t:
                for item in result:
                    print(title_index[str(item[0])])
        else:
            if not args.t and not args.v:
                newresult = rank_by_pr_score(pr_scores, result)
                print(' '.join([str(x) for x in newresult]))

            if args.t:
                newresult = rank_by_pr_score(pr_scores, result)
                for item in newresult:
                    print(title_index[str(item)])

            if args.v:
                newresult = rank_by_pr_score(pr_scores, result)
                prscore = {}
                file = open(pr_scores, 'r')
                for line in file:
                    line = line.strip()
                    docid, score = line.split('|')
                    prscore[int(docid)] = float(score)
                for item in newresult:
                    if item in prscore:
                        print(title_index[str(item)], ': ', prscore[item], sep='')
                    else:
                        print(title_index[str(item)], ': ', 0, sep='')
