import random
import tiktoken
from schema_validator import SchemaNode, StreamingJSONValidator, ObjSchema, ArrSchema

DEBUG = False  # set to True for debugging and JSON validation
INDENT_STEP = 2  # number of spaces for indentation

enc = tiktoken.get_encoding('gpt2')

TOK_QUOTE = enc.encode('"')[0]
TOK_COMMA = enc.encode(',')[0]
TOK_CLOSE_BRACKET = enc.encode(']')[0]

# ----------------------------------------------------------------------------
# 1.  Per–primitive allowed-token sets
# ----------------------------------------------------------------------------
TOKS_STR = {TOK_QUOTE} | {
    tid
    for tid, tok in enumerate(enc.decode([i]) for i in range(enc.n_vocab))
    if '"' not in tok and '\\' not in tok and '\n' not in tok
}

DIGITS = {
    tid
    for tid, tok in enumerate(enc.decode([i]) for i in range(enc.n_vocab))
    if all(ord('0') <= ord(c) <= ord('9') for c in tok)
}
TOK_DOT = enc.encode('.')[0]
TOKS_E = {enc.encode('e')[0], enc.encode('E')[0]}
TOKS_SIGN = {enc.encode('+')[0], enc.encode('-')[0]}

TOK_TRUE = enc.encode('true')[0]
TOK_FALSE = enc.encode('false')[0]
TOKS_BOOL = {TOK_TRUE, TOK_FALSE}

# ----------------------------------------------------------------------------
# 2.  The generic sampler  ----------------------------------------------------
rng = random.Random()  # expose/replace as needed


def sample(allowed_ids: set[int]) -> int:
    """Choose *one* token id from the allowed set."""
    return rng.choice(tuple(allowed_ids))


# ----------------------------------------------------------------------------
# 3.  Primitive samplers  -----------------------------------------------------
def sample_string(max_len: int = 12) -> str:
    """Return a complete JSON string literal, including the two quotes."""
    tids = [TOK_QUOTE]
    while len(tids) - 1 < max_len:  # minus the closing quote
        tids.append(sample(TOKS_STR))
        if tids[-1] == TOK_QUOTE:  # unescaped quote → done
            break
    else:  # max_len reached
        tids.append(TOK_QUOTE)
    return enc.decode(tids)


def sample_int(max_len: int = 8) -> str:
    tids = [sample(TOKS_SIGN | DIGITS)]
    while len(tids) < max_len:
        toks = DIGITS
        if tids and tids[-1] in DIGITS:
            toks |= {TOK_COMMA}  # comma token can finish
        tids.append(sample(toks))
        if tids[-1] == TOK_COMMA:
            tids.pop()  # don’t keep the comma
            break
    return enc.decode(tids)


def sample_float(max_len: int = 10) -> str:
    tids = [sample(TOKS_SIGN | DIGITS | {TOK_DOT} | TOKS_E)]

    def ends_with_digit():
        return enc.decode([tids[-1]])[-1] in '0123456789'

    while len(tids) < max_len or not ends_with_digit():
        toks = DIGITS
        if TOK_DOT not in tids:  # only one dot allowed
            toks |= {TOK_DOT}
        if not set(tids).intersection(TOKS_E):  # only one exponent allowed
            toks |= TOKS_E
        if tids and ends_with_digit():
            toks |= {TOK_COMMA}
        tids.append(sample(toks))
        if tids[-1] == TOK_COMMA:
            tids.pop()  # don’t keep the comma
            break

    return enc.decode(tids)


def sample_bool() -> str:
    return enc.decode([sample(TOKS_BOOL)])


def sample_null() -> str:
    return 'null'  # no randomness needed


PRIM_SAMPLER = {
    'str': sample_string,
    'int': sample_int,
    'float': sample_float,
    'bool': sample_bool,
    'null': sample_null,
}


# ----------------------------------------------------------------------------
# 4.  Streaming renderer (unchanged except for array length logic) -----------
def generate_json(schema, max_array_len: int = 4) -> str:
    out: list[str] = []
    if DEBUG:
        v = StreamingJSONValidator(schema)  # Just for validation - for sampling we don't need it

    indent = 0  # current indent level

    def emit(ch: str, with_indent=False):
        nonlocal indent
        if DEBUG:
            v.push(ch)

        if ch in '{[':  # opening → write char, newline, indent+1
            out.append(ch)
            indent += 1
            out.append('\n' + ' ' * (INDENT_STEP * indent))
            return

        if with_indent:  # used with a comma
            assert ch == ','
            out.append(ch)
            out.append('\n' + ' ' * (INDENT_STEP * indent))
            return

        if ch in '}]':  # closing → newline, indent-1, char
            indent -= 1
            out.append('\n' + ' ' * (INDENT_STEP * indent))
            out.append(ch)
            return

        out.append(ch)  # most tokens land here

    def walk(node: SchemaNode):
        if isinstance(node, ObjSchema):
            emit('{')
            first = True
            for k, sub in node.props.items():
                if not first:
                    emit(',', with_indent=True)
                first = False
                for ch in f'"{k}": ':
                    emit(ch)
                walk(sub)
            emit('}')
        elif isinstance(node, ArrSchema):
            emit('[')
            for i in range(max_array_len):
                walk(node.items)
                if sample({TOK_COMMA, TOK_CLOSE_BRACKET}) == TOK_CLOSE_BRACKET or i == max_array_len - 1:
                    break
                emit(',', with_indent=True)
            emit(']')
        else:  # primitive
            literal = PRIM_SAMPLER[node]()
            for ch in literal:
                emit(ch)

    walk(schema)
    if DEBUG:
        v.close()
    return ''.join(out)


# ----------------------------------------------------------------------------
# 5.  Quick demo --------------------------------------------------------------
if __name__ == '__main__':
    from pydantic import BaseModel
    from typing import List

    class Address(BaseModel):
        city: str
        zip: int

    class Person(BaseModel):
        name: str
        age: int
        address: Address
        tags: List[str]
        is_employed: bool

    person_schema = ObjSchema(
        {
            'name': 'str',
            'age': 'int',
            'address': ObjSchema({'city': 'str', 'zip': 'int'}),
            'tags': ArrSchema('str'),
            'is_employed': 'bool',
        }
    )

    # rng.seed(0)  # deterministic run
    j = generate_json(person_schema)
    print(j)  # → valid, random-looking JSON
