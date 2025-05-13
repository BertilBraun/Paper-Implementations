"""
Pydantic model  →  internal schema  →  grammar-constrained sampler → JSON text (valid for the schema)  →  Pydantic model instance
"""

from __future__ import annotations

import tiktoken
from pydantic import BaseModel
from typing import Any, Dict, List, Sequence, Tuple, get_args, get_origin

from schema_validator import StreamingJSONValidator, SchemaNode, ObjSchema, ArrSchema

# ──────────────────────────────────────────────────────────────────────────────
# 0 Tokenizer  (GPT-2 BPE)
# ──────────────────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding('gpt2')
ID2TOK: Dict[int, str] = {i: enc.decode([i]) for i in range(enc.n_vocab)}
TOK2ID: Dict[str, int] = {t: i for i, t in ID2TOK.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 1 Trie
# ──────────────────────────────────────────────────────────────────────────────
class TrieNode:
    __slots__ = ('kids', 'terminal')

    def __init__(self):
        self.kids: Dict[str, 'TrieNode'] = {}
        self.terminal: List[int] = []


def build_trie(id2tok: Dict[int, str]) -> TrieNode:
    root = TrieNode()
    for tid, tok in id2tok.items():
        node = root
        for ch in tok:
            node = node.kids.setdefault(ch, TrieNode())
        node.terminal.append(tid)
    return root


TRIE_ROOT = build_trie(ID2TOK)


# ──────────────────────────────────────────────────────────────────────────────
# 2 Trie-guided token filter
# ──────────────────────────────────────────────────────────────────────────────


def allowed_token_ids(pv: StreamingJSONValidator, root: TrieNode) -> List[int]:
    allowed: List[int] = []
    stack: List[Tuple[TrieNode, StreamingJSONValidator]] = [(root, pv)]
    tested = 0
    while stack:
        node, state = stack.pop()
        allowed.extend(node.terminal)
        for ch, child in node.kids.items():
            new_state = state.copy()
            tested += 1
            try:
                new_state.push(ch)
            except ValueError:
                continue
            stack.append((child, new_state))

    return allowed


# ──────────────────────────────────────────────────────────────────────────────
# 3 Sampler (dummy random logits)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np

RNG = np.random.default_rng(0)


def sample_json(schema: SchemaNode, max_steps: int) -> str:
    pv = StreamingJSONValidator(schema)
    out: List[str] = []

    for _ in range(max_steps):
        ids = allowed_token_ids(pv, TRIE_ROOT)
        tid = int(RNG.choice(ids))
        tok = ID2TOK[tid]
        out.append(tok)
        for ch in tok:
            pv.push(ch)
        if pv.done:
            break
    return ''.join(out)


# ──────────────────────────────────────────────────────────────────────────────
# 4 Pydantic integration
# ──────────────────────────────────────────────────────────────────────────────


def _schema_from_type(tp: Any) -> SchemaNode:
    """Recursively convert typing/Pydantic type → internal SchemaNode."""
    origin = get_origin(tp) or tp
    args = get_args(tp)

    # nested BaseModel
    if isinstance(origin, type) and issubclass(origin, BaseModel):
        return schema_from_pydantic(origin)

    # list / Sequence
    if origin in (list, List, Sequence):
        return ArrSchema(_schema_from_type(args[0]))

    # primitives
    if origin is str:
        return 'str'
    if origin is int:
        return 'int'
    if origin is float:
        return 'float'
    if origin is bool:
        return 'bool'
    if origin is type(None):
        return 'null'

    raise NotImplementedError(f'type {tp} not supported')


def schema_from_pydantic(model_cls: type[BaseModel]) -> ObjSchema:
    """Build an ObjSchema from a Pydantic BaseModel class (recursively)."""
    props: Dict[str, SchemaNode] = {
        name: _schema_from_type(field.annotation) for name, field in model_cls.model_fields.items()
    }
    return ObjSchema(props)


def instance_from_json(json_text: str, model_cls: type[BaseModel]):
    """Round-trip: parse JSON into a concrete Pydantic model instance."""
    return model_cls.model_validate_json(json_text)


if __name__ == '__main__':

    class Address(BaseModel):
        city: str
        zip: int

    class Person(BaseModel):
        name: str
        age: int
        address: Address
        tags: List[str]
        is_employed: bool

    person_schema = schema_from_pydantic(Person)
    print('SCHEMA:', person_schema)

    json_doc = sample_json(person_schema, max_steps=2000)
    print('GENERATED JSON:', json_doc)

    person = instance_from_json(json_doc, Person)
    print('\n⇢ Parsed back to Pydantic object\n', person)
