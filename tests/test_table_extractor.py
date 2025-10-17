import pytest
import os
from src.table_extractor import TableExtractor

# Dummy PDF for testing - in a real scenario, you'd create a small PDF with a table
# For now, we'll mock the camelot.read_pdf call.

class MockTable:
    def __init__(self, df):
        self.df = df

@pytest.fixture
def mock_camelot_read_pdf(monkeypatch):
    def mock_read_pdf(pdf_path, pages, flavor, suppress_stdout):
        if "dummy_pdf_with_table.pdf" in pdf_path:
            # Simulate a simple table
            df = [['Header1', 'Header2'], ['Row1Col1', 'Row1Col2'], ['Row2Col1', 'Row2Col2']]
            return [MockTable(df)]
        return []
    monkeypatch.setattr('camelot.read_pdf', mock_read_pdf)

def test_table_extraction_success(mock_camelot_read_pdf):
    extractor = TableExtractor()
    pdf_path = "dummy_pdf_with_table.pdf"
    tables = extractor.extract_tables(pdf_path)

    assert len(tables) == 1
    assert tables[0].iloc[0, 0] == 'Header1'
    assert tables[0].iloc[1, 1] == 'Row1Col2'

def test_table_extraction_no_tables(mock_camelot_read_pdf):
    extractor = TableExtractor()
    pdf_path = "dummy_pdf_without_tables.pdf"
    tables = extractor.extract_tables(pdf_path)

    assert len(tables) == 0

def test_table_extraction_error_handling(monkeypatch):
    def mock_read_pdf_error(pdf_path, pages, flavor, suppress_stdout):
        raise Exception("Simulated Camelot error")
    monkeypatch.setattr('camelot.read_pdf', mock_read_pdf_error)

    extractor = TableExtractor()
    pdf_path = "dummy_pdf_with_error.pdf"
    tables = extractor.extract_tables(pdf_path)

    assert len(tables) == 0
