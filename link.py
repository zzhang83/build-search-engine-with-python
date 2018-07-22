import json
import re
from argparse import ArgumentParser
from typing import Dict, List
from xml.etree import cElementTree
import numpy as np

def parse(collection_path: str) -> Dict[int, List[int]]:
    """Parses the collection file and returns a dictionary mapping
    documents to the documents that they link to.

    The dictionary keys and values are int document ids.

    Note: We recommend that you don't change this code.
    """
    root = cElementTree.parse(collection_path).getroot()
    match = re.match(r'{.*}', root.tag)
    namespace = match.group() if match else ''

    doc_ids = {}
    outlink_titles = {}
    for page in root.iter(namespace + 'page'):
        id_ = int(page.find(namespace + 'id').text)
        title = page.find(namespace + 'title').text
        assert id_ is not None and title is not None
        # Note this doesn't work on the small index, we aren't using
        # the small index anymore in the course
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
    """Returns the links in the body text. The links are
    title strings.

    Note: We recommend that you don't change this code.
    """
    return re.findall(r'\[\[([^\]|#]+)', text)


def get_isolates(outlinks: Dict[int, List[int]]) -> List[int]:
    """Returns all doc ids which have no inbound nor
    outbound links.

    Note: We recommend that you don't change this code.
    """
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


def main(collection_path: str):
    """Saves the outlinks dictionary as a JSON file then computes
    and saves the PageRank scores.

    Note: We recommend that you don't change this code.
    """
    try:
        with open('links.json', 'r') as fp:
            with_str_keys = json.load(fp)
            outlinks = {int(key): val for key, val in with_str_keys.items()}
        print('Using existing links file')
    except FileNotFoundError:
        print('Creating new links file')
        outlinks = parse(collection_path)
        with open('links.json', 'w') as fp:
            json.dump(outlinks, fp)

    scores = rank(outlinks)

    with open('scores.dat', 'w') as fp:
        for id_, score in sorted(list(scores.items()), key=lambda p: -p[1]):
            fp.write('{}|{}\n'.format(id_, score))


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
    assert rank(links) == {0: 0.1917313106536865, 1: 0.21526274406433107, 2: 0.13354674324035645, 3: 0.45945920204162594}
    assert rank(links2) == {0: 0.25, 1: 0.25, 7: 0.25, 12: 0.25}

def test_get_isolates():
    outlinks = {38075: [], 38076: [], 38077: [3843, 3842, 3843], 38080: [35650, 1463,2667], 38081: [3843, 3842, 3843], 38084: [25474, 34517, 37219, 27658], 38089: [25474, 37059, 38084], 38090: [16774], 38098: []}
    outlinks2 = {0: [1, 3], 1: [3], 2: [], 3: []}
    assert get_isolates(outlinks2) == [2]
    assert get_isolates(outlinks)== [38075, 38076, 38098]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('collection')
    args = parser.parse_args()
    main(args.collection)
