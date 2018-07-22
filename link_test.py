import pytest
from link import get_isolates, extract_links, rank


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


def test_get_isolates():
    links = {
        0: [1, 3],
        1: [3],
        2: [],
        3: [],
    }
    assert get_isolates(links) == [2]


def test_rank_simple():
    links = {
        0: [1, 3],
        1: [3],
        3: [],
    }
    assert pytest.approx(rank(links), {
        0: 0.19823767896947872,
        1: 0.2818420026899005,
        3: 0.5199203183406207
    })

def test_rank_symmetric():
    links = {
        0: [1, 3],
        1: [3, 8],
        3: [8, 0],
        8: [0, 1],
    }
    assert rank(links) == {0: 0.25, 1: 0.25, 3: 0.25, 8: 0.25}
