import pytest

from anfis_toolbox.membership import BellMF, GaussianMF, PiMF, SigmoidalMF, TrapezoidalMF, TriangularMF


@pytest.mark.parametrize(
    "mf",
    [
        GaussianMF(mean=0.0, sigma=1.0),
        TriangularMF(a=0.0, b=1.0, c=2.0),
        TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0),
        BellMF(a=0.5, b=1.0, c=2.0),
        SigmoidalMF(a=0.5, c=1.0),
        PiMF(a=0.1, b=1.0, c=2.0, d=3.0),
    ],
)
def test_str_repr(mf):
    s = str(mf)
    r = repr(mf)

    # Tipo correto
    assert isinstance(s, str)
    assert isinstance(r, str)

    # Contém o nome da classe e "MF"
    assert mf.__class__.__name__ in s
