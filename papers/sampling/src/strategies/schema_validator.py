from __future__ import annotations

"""
Incremental (stream‑oriented) JSON‑against‑schema validator
==========================================================

Feed *one character at a time* with :py:meth:`push`.  As soon as an input
byte contradicts either JSON syntax **or** the supplied schema, a
:class:`ValidationError` is raised.

The implementation purposefully supports just the essentials of RFC 8259
(e.g. strings are UTF‑8, numbers are *decimal* only).  Each validator is
a tiny FSM, so extending behaviour—scientific notation, comments,
optional keys—is local and independent.
"""

from dataclasses import dataclass
from typing import Mapping, Union, List, Optional, Literal

DEBUG = True  # set to False to disable output restrictions

# ── Schema model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObjSchema:
    props: Mapping[str, SchemaNode]


@dataclass(frozen=True)
class ArrSchema:
    items: SchemaNode


Primitive = Literal['str', 'int', 'float', 'bool', 'null']
SchemaNode = Union[ObjSchema, ArrSchema, Primitive]


# ── Validation core ───────────────────────────────────────────────────────────
class ValidationError(ValueError):
    def __init__(self, msg: str, path: List[str]):
        super().__init__(f"{'.'.join(path) or '<root>'}: {msg}")
        self.path = path[:]


def is_space(c: str) -> bool:
    """True if c is a space char."""
    return c in ' \t\n'


class Validator:
    """Abstract base for *incremental* validators."""

    __slots__ = ('_done', '_path')

    def __init__(self, path: Optional[List[str]] = None):
        self._done = False
        self._path: List[str] = path or []

    # helpers ---------------------------------------------------------------
    @property
    def done(self) -> bool:  # finished successfully
        return self._done

    def _ensure_active(self, c: str):
        if self._done:
            raise ValidationError('unexpected data after value', self._path)

    # subclasses must implement this ---------------------------------------
    def push(self, c: str):
        raise NotImplementedError

    def clone(self) -> Validator:
        raise NotImplementedError


# ── Primitive validators ─────────────────────────────────────────────────────
class ExactStringValidator(Validator):
    """Matches *exactly* the literal supplied at construction."""

    __slots__ = ('_target', '_idx')

    def __init__(self, literal: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target, self._idx = literal, 0

    def push(self, c: str):
        if is_space(c) and self._idx == 0:
            return
        self._ensure_active(c)
        want = self._target[self._idx]
        if c != want:
            raise ValidationError(f'expected {want!r}, got {c!r}', self._path)
        self._idx += 1
        if self._idx == len(self._target):
            self._done = True

    def clone(self) -> ExactStringValidator:
        new = ExactStringValidator(self._target, path=self._path[:])
        new._idx = self._idx
        new._done = self._done
        return new


class JSONStringValidator(Validator):
    """Enough of RFC 8259 string grammar for keys & values."""

    __slots__ = ('_state', '_esc', '_u_buf', '_out', 'raw')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = 'expect_open_quote'
        self._esc = False
        self._u_buf: List[str] | None = None
        self._out: List[str] = []
        self.raw: Optional[str] = None  # available when done

    def push(self, c: str):
        self._ensure_active(c)
        if self._state == 'expect_open_quote':
            if c != '"':
                raise ValidationError('expected opening quote', self._path)
            self._state = 'in_string'
            return

        if self._state == 'in_string':
            if self._esc:
                if c in '"\\/bfnrt':
                    self._out.append('\\' + c)
                    self._esc = False
                    return
                if c == 'u':
                    self._u_buf = []
                    self._esc = False
                    return
                raise ValidationError('invalid escape', self._path)

            if DEBUG and len(self._out) > 20 and c != '"':
                raise ValidationError('string too long', self._path)

            if self._u_buf is not None:
                if c.lower() not in '0123456789abcdef':
                    raise ValidationError('bad \\u escape', self._path)
                self._u_buf.append(c)
                if len(self._u_buf) == 4:
                    self._out.append('\\u' + ''.join(self._u_buf))
                    self._u_buf = None
                return

            if c == '\\':
                self._esc = True
                return
            if c == '"':
                self.raw = ''.join(self._out)
                self._done = True
                return
            if ord(c) < 0x20:
                raise ValidationError('control char in string', self._path)
            self._out.append(c)
            return

        raise ValidationError('internal string‑parser state', self._path)

    def clone(self) -> JSONStringValidator:
        new = JSONStringValidator(path=self._path[:])
        new._state = self._state
        new._esc = self._esc
        new._u_buf = self._u_buf[:] if self._u_buf else None
        new._out = self._out[:]
        new._done = self._done
        new.raw = self.raw
        return new


class JSONNumberValidator(Validator):
    __slots__ = (
        '_want_float',
        '_buf',
        '_allow_sign',
        '_allow_point',
        '_allow_exp',
        '_seen_digit',
    )

    def __init__(self, want_float: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._want_float = want_float
        self._buf: List[str] = []
        self._allow_sign = True
        self._allow_point = True
        self._allow_exp = True
        self._seen_digit = False

    def push(self, c: str):
        self._ensure_active(c)
        # terminators: don’t consume – let parent re‑feed
        if is_space(c) or c in ',}]':
            lit = ''.join(self._buf)
            if not self._seen_digit:
                raise ValidationError('expected digit', self._path)
            is_float = any(ch in lit for ch in '.eE')
            if is_float != self._want_float:
                raise ValidationError(f"value must be {'float' if self._want_float else 'int'}", self._path)
            self._done = True
            return 'rewind'

        if DEBUG and len(self._buf) > 20:
            raise ValidationError('number too long', self._path)

        if c in '+-' and self._allow_sign:
            self._allow_sign = False
            self._buf.append(c)
            return
        if c.isdigit():
            self._seen_digit = True
            self._allow_sign = False
            self._buf.append(c)
            return
        if c == '.' and self._allow_point:
            self._want_float = True
            self._allow_point = False
            self._allow_sign = False
            self._buf.append(c)
            return
        if c in 'eE' and self._allow_exp:
            self._allow_exp = False
            self._allow_sign = True
            self._allow_point = False
            self._buf.append(c)
            return
        raise ValidationError('malformed number', self._path)

    def clone(self) -> JSONNumberValidator:
        new = JSONNumberValidator(self._want_float, path=self._path[:])
        new._buf = self._buf[:]  # copy the list
        new._allow_sign = self._allow_sign
        new._allow_point = self._allow_point
        new._allow_exp = self._allow_exp
        new._seen_digit = self._seen_digit
        new._done = self._done
        return new


# ── Combinator helper ────────────────────────────────────────────────────────
class OneOfValidator(Validator):
    """Try candidates in parallel; whichever matches first wins."""

    __slots__ = ('_candidates', '_active')

    def __init__(self, candidates: List[Validator], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidates = candidates
        self._active: Optional[Validator] = None

    def push(self, c: str):
        self._ensure_active(c)

        # ── If a candidate has already been chosen ────────────────────────
        if self._active:
            res = self._active.push(c)
            if self._active.done:
                self._done = True
            return res

        # ── Try every candidate with this char ────────────────────────────
        survivors: List[Validator] = []
        for cand in self._candidates:
            try:
                cand.push(c)
                survivors.append(cand)
            except ValidationError:
                continue

        if not survivors:
            raise ValidationError('no option matched', self._path)

        self._candidates = survivors
        if len(survivors) == 1:
            self._active = survivors[0]
            if self._active.done:
                self._done = True
        # value not finished yet
        return

    def clone(self) -> OneOfValidator:
        new = OneOfValidator([], path=self._path[:])
        new._candidates = [c.clone() for c in self._candidates]
        new._active = self._active.clone() if self._active else None
        new._done = self._done
        return new


# ── Composite validators ────────────────────────────────────────────────────
class JSONArrayValidator(Validator):
    __slots__ = ('_itemschema', '_state', '_child', '_need_value')

    def __init__(self, itemschema: SchemaNode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._itemschema: SchemaNode = itemschema
        self._state = 'expect_open_bracket'
        self._child: Optional[Validator] = None
        self._need_value = True  # toggles after each comma

    def push(self, c: str):
        self._ensure_active(c)
        while True:
            # ── if a child is active, feed it first ────────────────────────
            if self._child and not self._child.done:
                res = self._child.push(c)
                if res == 'rewind':  # child finished on this very char
                    continue
                return
            if self._child and self._child.done:
                self._child = None
                self._need_value = False

            # ── main FSM ────────────────────────────────────────────────────
            if self._state == 'expect_open_bracket':
                if is_space(c):
                    return
                if c != '[':
                    raise ValidationError("expected '['", self._path)
                self._state = 'in_array'
                return

            if self._state == 'in_array':
                if is_space(c):
                    return
                if c == ']' and self._need_value:
                    self._done = True  # empty []
                    return
                if c == ']' and not self._need_value:
                    self._done = True
                    return
                if c == ',' and not self._need_value:
                    self._need_value = True
                    return
                if self._need_value:
                    self._child = make_validator(self._itemschema, self._path + ['<arr>'])
                    # loop again without consuming current char so child sees it
                    continue
                raise ValidationError('unexpected char in array', self._path)

    def clone(self) -> JSONArrayValidator:
        new = JSONArrayValidator(self._itemschema, path=self._path[:])
        new._state = self._state
        new._need_value = self._need_value
        new._child = self._child.clone() if self._child else None
        new._done = self._done
        return new


class JSONObjectValidator(Validator):
    __slots__ = (
        '_schema',
        '_state',
        '_child',
        '_need_key',
        '_consumed',
        '_pending_key',
    )

    def __init__(self, schema: ObjSchema, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema = schema
        self._state = 'expect_open_brace'
        self._child: Optional[Validator] = None
        self._need_key = True  # next thing must be a key unless object empty
        self._consumed: list[str] = []
        self._pending_key: Optional[str] = None

    # ── main push ---------------------------------------------------------
    def push(self, c: str):
        self._ensure_active(c)
        while True:
            # ── feed child first ──────────────────────────────────────────
            if self._child and not self._child.done:
                res = self._child.push(c)
                if res == 'rewind':
                    continue
                return
            if self._child and self._child.done:
                if self._state == 'parsing_key':
                    self._child = None
                    self._state = 'expect_colon'
                    continue
                if self._state == 'parsing_value':
                    self._child = None
                    self._need_key = False
                    if len(self._consumed) == len(self._schema.props):
                        self._state = 'expect_close_brace'
                    else:
                        self._state = 'in_object'
                    continue

            # ── FSM states ────────────────────────────────────────────────
            if self._state == 'expect_open_brace':
                if is_space(c):
                    return
                if c != '{':
                    raise ValidationError("expected '{'", self._path)
                self._state = 'in_object'
                return

            if self._state == 'expect_close_brace':
                if is_space(c):
                    return
                if c != '}':
                    raise ValidationError("expected '}'", self._path)
                self._done = True
                return

            if self._state == 'in_object':
                if is_space(c):
                    return
                if c in '{[}]':
                    raise ValidationError('unexpected brace in object', self._path)
                if c == ',' and not self._need_key:
                    self._need_key = True
                    return
                if self._need_key:
                    self._pending_key = list(self._schema.props.keys())[len(self._consumed)]
                    self._child = ExactStringValidator(
                        f'"{self._pending_key}"', self._path + [f'<key ({self._pending_key})>']
                    )
                    self._state = 'parsing_key'
                    continue
                raise ValidationError('unexpected char in object', self._path)

            if self._state == 'expect_colon':
                if is_space(c):
                    return
                if c != ':':
                    raise ValidationError("expected ':' after key", self._path)
                key = self._pending_key  # captured string literal
                assert key is not None
                if key not in self._schema.props:
                    raise ValidationError(f'unexpected key {key!r}', self._path)
                if key in self._consumed:
                    raise ValidationError(f'duplicate key {key!r}', self._path)
                self._consumed.append(key)
                self._pending_key = None
                self._state = 'want_value'
                return

            if self._state == 'want_value':
                if is_space(c):
                    return
                # first non‑space char begins the value – create child but *do not* consume char yet
                self._child = make_validator(self._schema.props[self._consumed[-1]], self._path + [self._consumed[-1]])
                self._state = 'parsing_value'
                continue  # loop so new child sees current char

        # end while

    def clone(self) -> JSONObjectValidator:
        new = JSONObjectValidator(self._schema, path=self._path[:])
        new._state = self._state
        new._need_key = self._need_key
        new._consumed = self._consumed[:]
        new._pending_key = self._pending_key
        new._child = self._child.clone() if self._child else None
        new._done = self._done
        return new


# ── factory -----------------------------------------------------------------


def make_validator(schema: SchemaNode, path: List[str]) -> Validator:
    if isinstance(schema, ObjSchema):
        return JSONObjectValidator(schema, path=path)
    if isinstance(schema, ArrSchema):
        return JSONArrayValidator(schema.items, path=path)
    if schema == 'str':
        return JSONStringValidator(path=path)
    if schema == 'int':
        return JSONNumberValidator(False, path=path)
    if schema == 'float':
        return JSONNumberValidator(True, path=path)
    if schema == 'bool':
        return OneOfValidator(
            [
                ExactStringValidator('true', path=path),
                ExactStringValidator('false', path=path),
            ],
            path=path,
        )
    if schema == 'null':
        return ExactStringValidator('null', path=path)
    raise TypeError(f'bad schema node: {schema!r}')


# ── public façade -----------------------------------------------------------
class StreamingJSONValidator:
    __slots__ = ('_top',)

    def __init__(self, schema: SchemaNode):
        self._top = make_validator(schema, [])

    @property
    def done(self) -> bool:
        return self._top.done

    def push(self, c: str):
        if self._top.done:
            raise ValidationError('trailing data after valid JSON', [])
        self._top.push(c)

    def close(self):
        if not self._top.done:
            raise ValidationError('truncated JSON', [])

    def copy(self) -> StreamingJSONValidator:
        other = object.__new__(StreamingJSONValidator)
        other._top = self._top.clone()
        return other


# ── mini‑demo ---------------------------------------------------------------
if __name__ == '__main__':
    schema = ObjSchema(props={'name': 'str', 'age': 'int'})
    good = '{ "name" : "Ada" , "age" : 42 }'
    bad = '{ "name" : "Ada" , "age" : "forty-two" }'

    for content in (good, bad):
        print('validating: ', content)
        v = StreamingJSONValidator(schema)
        for ch in content:
            try:
                v.push(ch)
            except ValidationError as e:
                print('✗ validation error: ', e)
                break
        else:
            v.close()
            print('✓ validated: ', content)
