import camelot
import pandas as pd

class TableExtractor:
    def __init__(self):
        pass

    def extract_tables(self, pdf_path, pages='all'):
        """
        Extracts tables from a PDF file using Camelot.

        Args:
            pdf_path (str): The path to the PDF file.
            pages (str): The pages to extract tables from (e.g., '1', '1,3', '1-3', 'all').

        Returns:
            list: A list of pandas DataFrames, each representing a table.
        """
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice', suppress_stdout=True)
            extracted_dfs = []
            for table in tables:
                extracted_dfs.append(table.df)
            return extracted_dfs
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
            return []

if __name__ == '__main__':
    # Example usage (requires a PDF file with tables)
    # You can replace 'path/to/your/pdf_with_tables.pdf' with an actual PDF file
    # For testing, you might want to create a dummy PDF with a table or use a known one.
    # For now, this will just demonstrate the class structure.
    extractor = TableExtractor()
    # tables = extractor.extract_tables('path/to/your/pdf_with_tables.pdf')
    # if tables:
    #     for i, df in enumerate(tables):
    #         print(f"Table {i+1}:")
    #         print(df)
    # else:
    #     print("No tables extracted.")
    print("TableExtractor class created. No example PDF provided for live testing.")
