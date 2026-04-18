"""DR-1 Episodic conservation — property test.

For any episodic record added to β buffer at time t, there exists
some t' in [t, t + tau_max] such that the record is consumed by a
DE. Validates framework spec §6.2 DR-1.

Skeleton version (S5.3): uses an in-memory fake beta buffer and
fake DE that consumes N records per execution. Real β
implementation lands S7+ alongside swap protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import given, settings
from hypothesis import strategies as st


@dataclass
class FakeBetaRecord:
    record_id: int
    consumed_by: str | None = None


@dataclass
class FakeBetaBuffer:
    records: list[FakeBetaRecord] = field(default_factory=list)

    def append(self, rid: int) -> None:
        self.records.append(FakeBetaRecord(record_id=rid))

    def consume(self, n: int, de_id: str) -> None:
        for rec in self.records:
            if rec.consumed_by is None and n > 0:
                rec.consumed_by = de_id
                n -= 1


@given(
    record_count=st.integers(min_value=1, max_value=100),
    batch_size=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=30, deadline=None)
def test_dr1_all_records_eventually_consumed(
    record_count: int, batch_size: int
) -> None:
    buf = FakeBetaBuffer()
    for i in range(record_count):
        buf.append(i)

    de_counter = 0
    unconsumed = [r for r in buf.records if r.consumed_by is None]
    while unconsumed:
        de_id = f"de-{de_counter:04d}"
        buf.consume(batch_size, de_id)
        de_counter += 1
        unconsumed = [r for r in buf.records if r.consumed_by is None]

    assert all(r.consumed_by is not None for r in buf.records)
    # tau_max proxy: number of DEs needed is bounded
    expected_max_des = (record_count + batch_size - 1) // batch_size
    assert de_counter <= expected_max_des
