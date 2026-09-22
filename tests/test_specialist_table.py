from mimir.council.table import SPECIALISTS, render_table
from mimir.models.specialist import SpecialistName


def test_every_specialist_is_a_row():
    assert set(SPECIALISTS) == set(SpecialistName)


def test_rows_carry_what_the_four_dicts_carried():
    row = SPECIALISTS[SpecialistName.KUBERNETES_INVESTIGATOR]
    assert row.capabilities and row.prompt and row.max_risk
    assert "kubernetes_investigator" in render_table()


def test_a_row_without_tools_reasons_from_what_it_is_given():
    assert SPECIALISTS[SpecialistName.SYNTHESIS].tools_only
