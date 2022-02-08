import asyncio
from datetime import datetime, timezone
import logging
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import OutOfBoundsDatetime, tslib
from pandas.core.internals import make_block
from pandas.core.internals.managers import BlockManager, form_blocks
import sentry_sdk
from sqlalchemy import Boolean, Column, DateTime, Integer
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.elements import Label

from athenian.api import metadata
from athenian.api.db import Database, DatabaseLike
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.persistentdata.models import Base as PerdataBase
from athenian.api.models.precomputed.models import GitHubBase as PrecomputedBase
from athenian.api.models.state.models import Base as StateBase
from athenian.api.to_object_arrays import to_object_arrays_split
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH


async def read_sql_query(sql: ClauseElement,
                         con: DatabaseLike,
                         columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                        MetadataBase, PerdataBase, PrecomputedBase, StateBase],
                         index: Optional[Union[str, Sequence[str]]] = None,
                         soft_limit: Optional[int] = None,
                         ) -> pd.DataFrame:
    """Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query.
    Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql     : SQLAlchemy query object to be executed.
    con     : async SQLAlchemy database engine.
    columns : list of the resulting columns names, column objects or the model if SELECT *
    index   : Name(s) of the index column(s).
    soft_limit
            : Load this number of rows at maximum.

    Returns
    -------
    DataFrame

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    try:
        data = await con.fetch_all(query=sql)
    except Exception as e:
        try:
            sql = str(sql)
        except Exception:
            sql = repr(sql)
        sql = textwrap.shorten(sql, MAX_SENTRY_STRING_LENGTH - 500)
        logging.getLogger("%s.read_sql_query" % metadata.__package__).error(
            "%s: %s; %s", type(e).__name__, e, sql)
        raise e from None
    if soft_limit is not None and len(data) > soft_limit:
        data = data[:soft_limit]
    return wrap_sql_query(data, columns, index)


def _create_block_manager_from_arrays(
    arrays_typed: List[np.ndarray],
    arrays_obj: np.ndarray,
    names_typed: List[str],
    names_obj: List[str],
    size: int,
) -> BlockManager:
    assert len(arrays_typed) == len(names_typed)
    assert len(arrays_obj) == len(names_obj)
    range_index = pd.RangeIndex(stop=size)
    typed_index = pd.Index(names_typed)
    blocks = form_blocks(arrays_typed, typed_index, [typed_index, range_index])
    blocks.append(make_block(arrays_obj, placement=np.arange(len(arrays_obj)) + len(arrays_typed)))
    return BlockManager(blocks, [pd.Index(names_typed + names_obj), range_index])


def wrap_sql_query(data: List[Sequence[Any]],
                   columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                  MetadataBase, StateBase],
                   index: Optional[Union[str, Sequence[str]]] = None,
                   ) -> pd.DataFrame:
    """Turn the fetched DB records to a pandas DataFrame."""
    try:
        columns[0]
    except TypeError:
        dt_columns = _extract_datetime_columns(columns.__table__.columns)
        int_columns = _extract_integer_columns(columns.__table__.columns)
        bool_columns = _extract_boolean_columns(columns.__table__.columns)
        columns = [c.name for c in columns.__table__.columns]
    else:
        dt_columns = _extract_datetime_columns(columns)
        int_columns = _extract_integer_columns(columns)
        bool_columns = _extract_boolean_columns(columns)
        columns = [(c.name if not isinstance(c, str) else c) for c in columns]
    typed_cols_indexes = []
    typed_cols_names = []
    obj_cols_indexes = []
    obj_cols_names = []
    for i, column in enumerate(columns):
        if column in dt_columns or column in int_columns or column in bool_columns:
            cols_indexes = typed_cols_indexes
            cols_names = typed_cols_names
        else:
            cols_indexes = obj_cols_indexes
            cols_names = obj_cols_names
        cols_indexes.append(i)
        cols_names.append(column)
    log = logging.getLogger(f"{metadata.__package__}.wrap_sql_query")
    # we used to have pd.DataFrame.from_records + bunch of convert_*() in relevant columns
    # the current approach is faster for several reasons:
    # 1. avoid an expensive copy of the object dtype columns in the BlockManager construction
    # 2. call tslib.array_to_datetime directly without surrounding Pandas bloat
    # 3. convert to int in the numpy domain and thus do not have to mess with indexes
    #
    # an ideal conversion would be loading columns directly from asyncpg but that requires
    # quite some changes in their internals
    with sentry_sdk.start_span(op="wrap_sql_query/convert", description=str(size := len(data))):
        data_typed, data_obj = to_object_arrays_split(data, typed_cols_indexes, obj_cols_indexes)
        converted_typed = []
        discard_mask = None
        for column, values in zip(typed_cols_names, data_typed):
            if column in dt_columns:
                converted_typed.append(_convert_datetime(values))
            elif column in int_columns:
                values, discarded = _convert_integer(values, column, int_columns[column], log)
                converted_typed.append(values)
                if discarded is not None:
                    if discard_mask is None:
                        discard_mask = np.zeros(len(data), dtype=bool)
                    discard_mask[discarded] = True
            elif column in bool_columns:
                converted_typed.append(values.astype(bool))
            else:
                raise AssertionError("impossible: typed columns are either dt or int")
        if discard_mask is not None:
            left = ~discard_mask
            size = left.sum()
            converted_typed = [arr[left] for arr in converted_typed]
            data_obj = data_obj[:, left]
    with sentry_sdk.start_span(op="wrap_sql_query/pd.DataFrame()", description=str(size)):
        block_mgr = _create_block_manager_from_arrays(
            converted_typed, data_obj, typed_cols_names, obj_cols_names, size)
        frame = pd.DataFrame(block_mgr, columns=typed_cols_names + obj_cols_names, copy=False)
        for column in dt_columns:
            try:
                frame[column] = frame[column].dt.tz_localize(timezone.utc)
            except (AttributeError, TypeError):
                continue
        if index is not None:
            frame.set_index(index, inplace=True)
    return frame


def _extract_datetime_columns(columns: Iterable[Union[Column, str]]) -> Set[str]:
    return {
        c.name for c in columns
        if not isinstance(c, str) and (
            isinstance(c.type, DateTime) or
            (isinstance(c.type, type) and issubclass(c.type, DateTime))
        )
    }


def _extract_boolean_columns(columns: Iterable[Union[Column, str]]) -> Set[str]:
    return {
        c.name for c in columns
        if not isinstance(c, str) and (
            isinstance(c.type, Boolean) or
            (isinstance(c.type, type) and issubclass(c.type, Boolean))
        )
    }


def _extract_integer_columns(columns: Iterable[Union[Column, str]],
                             ) -> Dict[str, bool]:
    return {
        c.name: getattr(
            c, "info", {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
        ).get("erase_nulls", False)
        for c in columns
        if not isinstance(c, str) and (
            isinstance(c.type, Integer) or
            (isinstance(c.type, type) and issubclass(c.type, Integer))
        )
        and not getattr(c, "nullable", False)
        and (not isinstance(c, Label) or (
            (not getattr(c.element, "nullable", False))
            and (not getattr(c, "nullable", False))
        ))
    }


def _convert_datetime(arr: np.ndarray) -> np.ndarray:
    # None converts to NaT
    try:
        ts, offset = tslib.array_to_datetime(arr, utc=True, errors="raise")
        assert offset is None
    except OutOfBoundsDatetime:
        # TODO(vmarkovtsev): copy the function and set OOB values to NaT
        # this comparison is very slow but still faster than removing tzinfo and taking np.array()
        arr[arr == datetime(1, 1, 1)] = None
        arr[arr == datetime(1, 1, 1, tzinfo=timezone.utc)] = None
        try:
            return _convert_datetime(arr)
        except OutOfBoundsDatetime as e:
            raise e from None
    # 0 converts to 1970-01-01T00:00:00
    ts[ts == np.zeros(1, ts.dtype)[0]] = None
    return ts


def postprocess_datetime(frame: pd.DataFrame,
                         columns: Optional[Iterable[str]] = None,
                         ) -> pd.DataFrame:
    """Ensure *inplace* that all the timestamps inside the dataframe are valid UTC or NaT.

    :return: Fixed dataframe - the same instance as `frame`.
    """
    utc_dt1 = datetime(1, 1, 1, tzinfo=timezone.utc)
    dt1 = datetime(1, 1, 1)
    if columns is not None:
        obj_cols = dt_cols = columns
    else:
        obj_cols = frame.select_dtypes(include=[object])
        dt_cols = frame.select_dtypes(include=["datetime"])
    for col in obj_cols:
        fc = frame[col]
        if utc_dt1 in fc:
            fc.replace(utc_dt1, pd.NaT, inplace=True)
        if dt1 in fc:
            fc.replace(dt1, pd.NaT, inplace=True)
    for col in dt_cols:
        fc = frame[col]
        if 0 in fc:
            fc.replace(0, pd.NaT, inplace=True)
        try:
            frame[col] = fc.dt.tz_localize(timezone.utc)
        except (AttributeError, TypeError):
            continue
    return frame


def _convert_integer(arr: np.ndarray,
                     name: str,
                     erase_null: bool,
                     log: logging.Logger,
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    nulls = None
    while True:
        try:
            return arr.astype(int), nulls
        except TypeError as e:
            nulls = np.equal(arr, None)
            if not nulls.any() or not erase_null:
                raise ValueError(f"Column {name} is not all-integer") from e
            log.error("fetched nulls instead of integers in %s", name)
            arr[nulls] = 0


def postprocess_integer(frame: pd.DataFrame, columns: Iterable[Tuple[str, int]]) -> pd.DataFrame:
    """Ensure *inplace* that all the integers inside the dataframe are not objects.

    :return: Fixed dataframe, a potentially different instance.
    """
    dirty_index = False
    log = None
    for col, erase_null in columns:
        while True:
            try:
                frame[col] = frame[col].astype(int, copy=False)
                break
            except TypeError as e:
                nulls = frame[col].isnull().values
                if not nulls.any():
                    raise ValueError(f"Column {col} is not all-integer") from e
                if not erase_null:
                    raise ValueError(f"Column {col} is not all-integer\n"
                                     f"{frame.loc[nulls].to_dict('records')}") from e
                if log is None:
                    log = logging.getLogger(f"{metadata.__package__}.read_sql_query")
                log.error("fetched nulls instead of integers in %s: %s",
                          col, frame.loc[nulls].to_dict("records"))
                frame = frame.take(np.flatnonzero(~nulls))
                dirty_index = True
    if dirty_index:
        frame.reset_index(drop=True, inplace=True)
    return frame


async def gather(*coros_or_futures,
                 op: Optional[str] = None,
                 description: Optional[str] = None,
                 catch: Type[BaseException] = Exception,
                 ) -> Tuple[Any, ...]:
    """Return a future aggregating results/exceptions from the given coroutines/futures.

    This is equivalent to `asyncio.gather(*coros_or_futures, return_exceptions=True)` with
    subsequent exception forwarding.

    :param op: Wrap the execution in a Sentry span with this `op`.
    :param description: Sentry span description.
    :param catch: Forward exceptions of this type.
    """
    async def body():
        if len(coros_or_futures) == 0:
            return tuple()
        if len(coros_or_futures) == 1:
            return (await coros_or_futures[0],)
        results = await asyncio.gather(*coros_or_futures, return_exceptions=True)
        for r in results:
            if isinstance(r, catch):
                raise r from None
        return results

    if op is not None:
        with sentry_sdk.start_span(op=op, description=description):
            return await body()
    return await body()


async def read_sql_query_with_join_collapse(
        query: ClauseElement,
        db: Database,
        columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                       MetadataBase, PerdataBase, PrecomputedBase, StateBase],
        index: Optional[Union[str, Sequence[str]]] = None,
        soft_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Enforce the predefined JOIN order in read_sql_query()."""
    query = query.with_statement_hint("Set(join_collapse_limit 1)")
    return await read_sql_query(query, db, columns=columns, index=index, soft_limit=soft_limit)


# Allow other coroutines to execute every Nth iteration in long loops
COROUTINE_YIELD_EVERY_ITER = 250


async def list_with_yield(iterable: Iterable[Any], sentry_op: str) -> List[Any]:
    """Drain an iterable to a list, tracing the loop in Sentry and respecting other coroutines."""
    with sentry_sdk.start_span(op=sentry_op) as span:
        things = []
        for i, thing in enumerate(iterable):
            if (i + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                await asyncio.sleep(0)
            things.append(thing)
        try:
            span.description = str(i)
        except UnboundLocalError:
            pass
    return things
