import sys
import os
from bs4 import BeautifulSoup as bs
from nltk.stem.porter import *
import xml.etree.ElementTree as ET
from xml.etree import cElementTree
import numpy as np
from typing import Dict, List
import pytest

def main():
    """
    Generates index file in the form "sorted string|original" string each on their own line.
    Could've generated the dictionary of the form "[sorted string,anagram|anagram|anagram|]" but the
    spec seemed to imply that we shouldn't do that.
    :return: None
    """

    if len(sys.argv) < 4 or len(sys.argv) > 4:
        print('Error: The option to specify index files was removed in version 3a please input an index directory instead\n'
              'This program requires 3 arguments to run properly:\n'
              '1. stop_word_collection\n'
              '2. collection_of_pages\n'
              '3. file directory/')
        exit()
    else:
        stop_word_collection = sys.argv[1]
        collection_of_pages = sys.argv[2]
        index_folder = sys.argv[3]

    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
    try:
        tree = ET.parse(collection_of_pages)
    except ET.ParseError:  # catch the ctrl-D exit
        print('ERROR: Bad XML File encountered. Exiting.')
        exit()

    stop_word_collection_list = stop_list_generator(stop_word_collection)
    total_docs, mpi_output = parse_collection(collection_of_pages,index_folder, stop_word_collection_list)

    with open('./' + str(index_folder) + 'inverted_index.dat', 'w') as mpi:
        mpi.write('('+str(total_docs)+', '+str(mpi_output)+')')


def parse_collection(collection_of_pages,index_folder,stop_word_collection_list):
    """
    :param collection_of_pages:
    :return:
    """
    # beautiful soup the XML. Because the namespace note wasn't out when I started this.
    soup = bs(open(collection_of_pages), "xml")

    # initialize index dictionary
    master_positional_index_dict = {}

    # calculate PageRank
    outlinks = parse(collection_of_pages)
    scores = rank(outlinks)
    with open('./' + str(index_folder) + 'scores.dat', 'w') as fp:
        for id_, score in sorted(list(scores.items()), key=lambda p: -p[1]):
            fp.write('{}|{}\n'.format(id_, score))

    total_docs = len(soup.find_all('page'))

    # open two files
    with open('./' + str(index_folder) + 'title_index.dat', 'w') as dest_index:
        # traverse XML tree (grab all pages)
        for page in soup.find_all('page'):

            # grab relevant information
            title = page.title.string
            id_no = int(page.id.string)

            # bs has some reserved words for grabbing text which made grabbing the STRING in the TEXT node tough.
            # probably some syntax for that. didn't know it. this seems to work.
            text_body = page.select('text')[0].string

            # write un-processed titles to file
            dest_index.write('{}|{}\n'.format(id_no, title))

            # begin processing for positional index

            # regex for grabbing words uppercase ignored as we'll be "lowering"
            split_regex = re.compile(r'[^a-z0-9]')
            title = split_regex.split(title.lower())

            # catch blank text_body - but maintain addition as title may not be blank
            # maybe you should add a catch for blank titles to remove them (but assignment says nothing about this)
            if text_body is not None:
                text_body = split_regex.split(text_body.lower())
            else:
                text_body = ''

            # stemming
            stemmed_filtered_title = stop_rm_stem(title, stop_word_collection_list)
            stemmed_filtered_text = stop_rm_stem(text_body, stop_word_collection_list)

            # generate master positional index in the following form
            # {'WORD':[QTY,{DOC1:[QTY,[LOC,LOC,LOC],DocLength, PageRank],DOC2:[QTY,[LOC,LOC,LOC],DocLength, PageRank],DOC3:[QTY,[LOC,LOC,LOC],DocLength,PageRank]})]

            # get the number of words in the processed document
            term_increment = 1/len(stemmed_filtered_title+stemmed_filtered_text)
            document_length = len(stemmed_filtered_title+stemmed_filtered_text)

            # put stemmed/stop-word-removed title and text together
            # go through each word of the combination and keep track of position
            # NOTE: title will be included in positioning

            for i, word in enumerate(stemmed_filtered_title+stemmed_filtered_text):

                # Check if the word is in the index
                if word in master_positional_index_dict:

                    # store positional index for later use
                    temp_positional_list = master_positional_index_dict[word][1].get(id_no)


                    # if the word is in the index, check to see if it has been seen in this document
                    if temp_positional_list is None:

                        # if it hasn't been seen in this document add it in and add in document length

                        # {'WORD': [QTY, {DOC1: [QTY, [LOC, LOC, LOC], DocLength, PageRank]}
                        master_positional_index_dict[word][1][id_no] = [1, [i], document_length,scores.get(id_no,0)]
                        # the word is already present, but this is the first time it has been seen in this document
                        # therefore increase the document frequency count by one
                        master_positional_index_dict[word][0] += 1

                    else:
                        # if it HAS been seen in this document already, add this occurrence to the list and update freqs
                        master_positional_index_dict[word][1][id_no][1].append(i)
                        master_positional_index_dict[word][1][id_no][0] += 1

                # if it's not already in the the index, add it and add occurrence, and tack on document length.
                else:
                    master_positional_index_dict[word] = [1, {id_no: [1, [i], document_length,scores.get(id_no,0)]}]

        return total_docs, master_positional_index_dict



def stop_list_generator(stop_word_collection):
    """
    Reads and generates stop word list
    :param stop_word_collection:
    :return stop_word_list
    """
    # read stop words into list
    stop_word_list = []
    with open(stop_word_collection) as stop_word_read:

        # read first line
        source_line = stop_word_read.readline()

        while source_line:
            stop_word_list.append(source_line.strip('\n'))

            # move to next line
            source_line = stop_word_read.readline()

    stop_word_list.append('')  # this is a little hacky to address some issue.

    return stop_word_list


def stop_rm_stem(text_input, stop_word_collection_list, verbose=0):
    """
    Reads and generates stop word list
    :param stop_word_collection:
    :return stop_word_list
    """
    stemmer = PorterStemmer('NLTK_EXTENSIONS')

    not_a_stop_word_list = []

    if verbose == 0:
        not_a_stop_word_list = [word for word in text_input if word not in stop_word_collection_list]
    else:
        for word in text_input:
            if word not in stop_word_collection_list:
                not_a_stop_word_list.append(word)
            else:
                if word == '':
                    continue
                else:
                    print('FYI: \'{}\' is a stop word and is being removed from your query'.format(word))

    return [stemmer.stem(word) for word in not_a_stop_word_list]


def parse(collection_path: str) -> Dict[int, List[int]]:
    root = cElementTree.parse(collection_path).getroot()
    match = re.match(r'{.*}', root.tag)
    namespace = match.group() if match else ''

    doc_ids = {}
    outlink_titles = {}
    for page in root.iter(namespace + 'page'):
        id_ = int(page.find(namespace + 'id').text)
        title = page.find(namespace + 'title').text
        assert id_ is not None and title is not None

        text = page.find(namespace + 'revision').find(namespace + 'text').text
        if text is None:
            links = []
        else:
            links = extract_links(text)

        doc_ids[title] = id_
        outlink_titles[id_] = links

    outlink_ids = {}
    for id_, titles in outlink_titles.items():
        outlink_ids[id_] = [doc_ids[title]
                            for title in titles
                            if title in doc_ids]

    for id_ in get_isolates(outlink_ids):
        outlink_ids.pop(id_)

    return outlink_ids


def extract_links(text: str) -> List[str]:
    return re.findall(r'\[\[([^\]|#]+)', text)


def get_isolates(outlinks: Dict[int, List[int]]) -> List[int]:
    connected_ids = set()
    for id_, linked_ids in outlinks.items():
        if linked_ids:
            connected_ids.add(id_)
            connected_ids.update(linked_ids)

    return [id_ for id_ in outlinks if id_ not in connected_ids]


def rank(outlinks: Dict[int, List[int]],
         eps: float = 0.01,
         d: float = 0.85) -> Dict[int, float]:
    """Returns the PageRank scores of the documents stored in
    outlinks.

    :param outlinks Mapping of doc ids to the ids that they link to
    :param eps The convergence threshold
    :param d The damping factor
    """

    # TIMING ----- Pixar timed @ 2s ------------
    # start_time = time.time()
    # TIMING -----------------------------------

    N = len(outlinks)  # how many pages?
    M = np.zeros((N, N))  # initialize matrix

    # there are ~14k pages with ID's ranging from 6 to ~33k.  need to effectively reindex to keep
    # track of things

    matrix_mapping = sorted(outlinks.keys())  # first get a sorted list of keys

    matrix_mapped_dict = {}

    # here we are starting at 0 (because the matrix requires this indexing) and keeping track of
    # which page ID is associated with the first input, or 0.  i.e. the first page ID is 6, which
    # will be remapped to 0.  The next one, ~996, to 1, etc ...
    for location, key in enumerate(matrix_mapping):
        matrix_mapped_dict[key] = location

    # now we can use another simple counter to move through the matrix and set values along the way.
    # unfortunately at this point we will need to traverse each page in the parsed list
    for column_no, (page, outbound_links) in enumerate(outlinks.items()):

        # if the page has some outbound links we need to use the lookup dictionary made above
        # to re-map to the new index, use set operation to deduplicate, recast as list for use later)
        if len(outbound_links) != 0:
            converted_links = list(set([matrix_mapped_dict[link] for link in outbound_links]))
            # "converted_links" is a list containing the rows that need to be set
            # counter_2 indicates the current column
            M[converted_links, column_no] = 1/len(converted_links)
            # num_outbound[counter_1] = 1 / len(set((outbound_links)))
        else:  # address the sink. treat as links to all pages equally as per TA instruction.
            M[:, column_no] += 1 / N  # no outbound links? set all values of column to 1/N.

    page_rank_score_vector = np.full((N, 1), 1 / N)  # initialize first pass at scores.
    delta = 10000  # set delta to something absurdly high to ensure it runs.

    # iteratively calculate new scores until predetermined epsilon threshold is met.
    while delta > eps:
        t_0 = page_rank_score_vector

        # NOTE: Why is M*page_rank_score_vector different from matmul the same thing?
        page_rank_score_vector = d * np.matmul(M, page_rank_score_vector) + (1 - d) / N

        delta = np.linalg.norm(page_rank_score_vector - t_0)  # |t_1 - t_0| < eps  <--- threshold calc.

    # page_rank_vector is still re-indexed.  We'll need to switch it back to the original index.
    page_rank = {}

    # so ... again ... the original "index" was something like 6,963,965,etc..
    # for easy matrix manipulation we has to set that to 0,1,2,etc...
    # but when joining this back the main index / collection we need the original ID's.
    # above we created an ORDERED list and a dictionary based off that to map 6:0, 963:1, 965:2, etc...
    # we are now undoing those changes.  This works because vectors maintain order.  First row is page
    # 6's value, second row 963, etc...

    for location_tracker, item in enumerate(page_rank_score_vector):
        page_rank[matrix_mapping[location_tracker]] = item.tolist()[0]

    # TIMING ----- Pixar timed @ 2s ------------
    # print("Took: %s seconds to run." % (time.time() - start_time))
    # TIMING -----------------------------------

    return page_rank

def test_rank():
    links = {
        0: [1, 3],
        1: [3],
        2:[0,3],
        3: [],
    }
    links2 = {
        0: [1, 7],
        1: [7, 12],
        7: [12, 0],
        12: [0, 1],
    }
    assert pytest.approx(rank(links)) == {0: 0.1917313106536865, 1: 0.21526274406433107, 2: 0.13354674324035645, 3: 0.45945920204162594}
    assert rank(links2) == {0: 0.25, 1: 0.25, 7: 0.25, 12: 0.25}


def test_get_isolates():
    outlinks = {38075: [], 38076: [], 38077: [3843, 3842, 3843], 38080: [35650, 1463,2667], 38081: [3843, 3842, 3843], 38084: [25474, 34517, 37219, 27658], 38089: [25474, 37059, 38084], 38090: [16774], 38098: []}
    outlinks2 = {0: [1, 3],1: [3],2: [],3: []}
    assert get_isolates(outlinks2) == [2]
    assert get_isolates(outlinks)== [38075, 38076, 38098]


def test_extract_links_basic():
    text = 'abc [[Nemo]]'
    assert extract_links(text) == ['Nemo']

def test_extract_links_bar():
    text = 'abc [[Character:Nemo|Nemo]] efg [[Character:Dory|Dory]]'
    assert extract_links(text) == ['Character:Nemo', 'Character:Dory']

def test_extract_links_hash():
    text = 'abc [[Character:Nemo|Nemo]] efg [[Character:Dory#Childhood]]'
    assert extract_links(text) == ['Character:Nemo', 'Character:Dory']

def test_extract_links_empty():
    text = ''
    assert extract_links(text) == []


def test_stop_word_removal_and_lem():
    stop_words = ['one', 'two', '123']
    assert stop_rm_stem(['one'], stop_words) == []
    assert stop_rm_stem(['tWo'], stop_words) == ['two']
    assert stop_rm_stem(['tWo', 'one'], stop_words) == ['two']
    assert stop_rm_stem(['tWo'], stop_words) == ['two']
    assert stop_rm_stem(['123', 'dog', 'cat'], stop_words) == ['dog', 'cat']
    assert stop_rm_stem(['dory'], stop_words) == ['dori']
    assert stop_rm_stem(['dory908049834023dog'], stop_words) == ['dory908049834023dog']
    assert stop_rm_stem(["dory908049834023dog", "%#$#", "fsd", "(("], stop_words) == ["dory908049834023dog",
                                                                                      "%#$#", "fsd", "(("]

if __name__ == '__main__':
    main()

