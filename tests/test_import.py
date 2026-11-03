"""パッケージが読み込めることを確かめる（最初のテストができたら消してよい）。"""

import soccer_xg_vs_llm


def test_import_package_has_docstring() -> None:
    """パッケージを読み込むと docstring がある。"""
    assert soccer_xg_vs_llm.__doc__
