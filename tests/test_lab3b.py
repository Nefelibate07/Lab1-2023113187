import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Lab1 import TextGraphLab

@pytest.fixture
def app():
    g = TextGraphLab()
    words = ["a", "b", "c", "a", "d", "e", "b", "d"]
    g.build_graph(words)
    return g

def test_bridge_exists(app):
    # a -> b -> c，所以 a 到 c 的桥接词是 b
    assert app.query_bridge_words("a", "c") == 'The bridge word from "a" to "c" is: b.'

def test_no_bridge_words(app):
    # c 到 e 没有桥接词
    assert app.query_bridge_words("c", "e") == 'No bridge words from "c" to "e"!'

def test_first_word_not_exist(app):
    assert app.query_bridge_words("x", "a") == 'No "x" in the graph!'

def test_second_word_not_exist(app):
    assert app.query_bridge_words("a", "x") == 'No "x" in the graph!'

def test_both_words_not_exist(app):
    assert app.query_bridge_words("x", "y") == 'No "x" and "y" in the graph!'

def test_case_insensitive(app):
    assert app.query_bridge_words("A", "C") == 'The bridge word from "a" to "c" is: b.'