"""Official CodeChameleon encryption rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

EncryptedQuery: TypeAlias = str | list[dict[str, int]] | dict[str, object] | list[str]


@dataclass
class _TreeNode:
    value: str
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


def encrypt_binary_tree(sentence: str) -> dict[str, object] | None:
    """Encrypt by arranging words into the official balanced binary tree."""

    def build_tree(words: list[str], start: int, end: int) -> _TreeNode | None:
        if start > end:
            return None
        mid = (start + end) // 2
        node = _TreeNode(words[mid])
        node.left = build_tree(words, start, mid - 1)
        node.right = build_tree(words, mid + 1, end)
        return node

    def tree_to_json(node: _TreeNode | None) -> dict[str, object] | None:
        if node is None:
            return None
        return {
            "value": node.value,
            "left": tree_to_json(node.left),
            "right": tree_to_json(node.right),
        }

    words = sentence.split()
    root = build_tree(words, 0, len(words) - 1)
    return tree_to_json(root)


def encrypt_reverse(sentence: str) -> str:
    """Encrypt by reversing space-separated tokens, matching official code."""
    return " ".join(sentence.split(" ")[::-1])


def encrypt_none(sentence: str) -> str:
    """Leave the query unchanged."""
    return sentence


def encrypt_odd_even(sentence: str) -> str:
    """Encrypt by placing odd-indexed words before even-indexed words."""
    words = sentence.split()
    odd_words = words[::2]
    even_words = words[1::2]
    return " ".join(odd_words + even_words)


def encrypt_length(sentence: str) -> list[dict[str, int]]:
    """Encrypt by sorting words by length while preserving original indices."""

    @dataclass
    class WordData:
        word: str
        index: int

    word_data = [WordData(word, i) for i, word in enumerate(sentence.split())]
    word_data.sort(key=lambda x: len(x.word))
    return [{data.word: data.index} for data in word_data]


def get_encrypted_query(sentence: str, encrypt_rule: str) -> EncryptedQuery:
    """Encrypt one query using the official rule names."""
    if encrypt_rule == "none":
        return encrypt_none(sentence)
    if encrypt_rule == "binary_tree":
        tree = encrypt_binary_tree(sentence)
        return tree if tree is not None else {}
    if encrypt_rule == "reverse":
        return encrypt_reverse(sentence)
    if encrypt_rule == "odd_even":
        return encrypt_odd_even(sentence)
    if encrypt_rule == "length":
        return encrypt_length(sentence)
    return ["Error: Encrypt rule is invalid. Please provide a correct encrypt rule."]
