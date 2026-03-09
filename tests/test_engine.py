"""
Engine smoke tests for DoseTrack V4.
Runs without Streamlit — tests the PK/PD simulation directly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
import sqlite3
import tempfile

from dosetrack import simulate, Dose


def test_single_dose():
    result = simulate(
        doses=[Dose(time_h=0.0, amount_mg=50.0)],
        weight_kg=70.0,
        model="prodrug",
        t_span=(0.0, 30.0),
    )
    assert result.peak_effect > 0, "Peak effect should be positive"
    assert result.peak_effect <= 100, "Peak effect should not exceed 100%"
    assert float(result.effect_pct[0]) == 0.0, "Effect at t=0 should be 0 (drug not yet absorbed)"
    assert result.tmax_h > 0, "Tmax should be after dosing"
    print(f"  Single dose: peak={result.peak_effect:.1f}% at t={result.tmax_effect_h:.1f}h")


def test_two_doses_cumulative():
    result = simulate(
        doses=[Dose(time_h=0.0, amount_mg=50.0), Dose(time_h=6.0, amount_mg=20.0)],
        weight_kg=70.0,
        model="prodrug",
        t_span=(0.0, 30.0),
    )
    assert result.peak_effect > 0
    assert len(result.t) > 100
    print(f"  Two doses: peak={result.peak_effect:.1f}%")


def test_da_stores_deplete():
    result = simulate(
        doses=[Dose(time_h=0.0, amount_mg=70.0)],
        weight_kg=70.0,
        model="prodrug",
        t_span=(0.0, 30.0),
    )
    min_da = float(result.da_stores.min())
    assert min_da < 1.0, "DA stores should deplete during dosing"
    print(f"  DA stores: min={min_da:.2f}")


def test_db_operations():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS doses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            dt TEXT NOT NULL,
            mg REAL NOT NULL
        )
    """)
    con.commit()

    con.execute("INSERT INTO doses (username, dt, mg) VALUES (?, ?, ?)",
                ("test", datetime(2026, 3, 9, 8, 0).isoformat(), 50.0))
    con.commit()

    rows = con.execute("SELECT id, dt, mg FROM doses WHERE username=?", ("test",)).fetchall()
    assert len(rows) == 1
    assert rows[0][2] == 50.0

    con.execute("DELETE FROM doses WHERE username=?", ("test",))
    con.commit()

    rows = con.execute("SELECT id FROM doses WHERE username=?", ("test",)).fetchall()
    assert len(rows) == 0
    con.close()
    os.unlink(db_path)
    print("  DB operations: OK")


if __name__ == "__main__":
    print("Running DoseTrack V4 engine tests...")
    test_single_dose()
    test_two_doses_cumulative()
    test_da_stores_deplete()
    test_db_operations()
    print("\nAll tests passed.")
