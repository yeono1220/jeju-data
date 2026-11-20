

def test_synthetic_report_marks_every_page(tmp_path):
    """가상데이터 보고서는 쪽마다 표시가 붙는다(실제 자료 보고서에는 없다)."""
    from dataolle.report.document import Report, to_html
    a = to_html(Report(title="t", subtitle="s", synthetic=True), tmp_path / "a.html").read_text(encoding="utf-8")
    b = to_html(Report(title="t", subtitle="s", synthetic=False), tmp_path / "b.html").read_text(encoding="utf-8")
    assert "@top-right" in a and "@top-right" not in b
