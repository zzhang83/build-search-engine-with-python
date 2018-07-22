import pytest

from wikiasearch.boolparser import parse_boolean


def test_single_word():
    assert parse_boolean('Hello') == 'Hello'


def test_keep_non_alphanumeric():
    assert parse_boolean('He?ll?o') == 'He?ll?o'


def test_single_parentheses():
    assert parse_boolean('(HELLO)') == '(HELLO)'


def test_operator_precedence():
    expected = ('OR', ['Hello', ('AND', ['WORLD', '!'])])
    assert parse_boolean('Hello OR (WORLD) AND !') == expected


def test_no_unnecessary_nesting():
    expected = ('AND', ['Hello', 'WORLD', '!'])
    assert parse_boolean('(Hello AND (WORLD) AND !)') == expected


def test_precedence_parentheses():
    expected = ('AND', [('OR', ['Hello', 'WORLD']), '!'])
    assert parse_boolean('(Hello OR (WORLD)) AND !') == expected


def test_empty_string():
    with pytest.raises(ValueError):
        parse_boolean('')


def test_whitespace_only():
    with pytest.raises(ValueError):
        parse_boolean('  ')


def test_and_or():
    with pytest.raises(ValueError):
        parse_boolean('AND OR')


def test_ending_and():
    with pytest.raises(ValueError):
        parse_boolean('(nemo AND fish) AND')


def test_ending_and_whitespace():
    with pytest.raises(ValueError):
        parse_boolean('(nemo AND fish) AND  ')
