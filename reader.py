# import tabula
import PyPDF2 
# from tabulate import tabulate 
# import fitz 
# import io 
from typing import List 
# from pdf2docx import Converter 


def extract_text(file_path: str):
    parsed_pdf = PyPDF2.PdfReader(file_path)
    texts = []
    for page_number in range(0, len(parsed_pdf.pages)):
        pageObj = parsed_pdf.pages[page_number] 
        text = pageObj.extract_text() 
        texts.append(text)
    return ''.join(texts)

def text_extractor(path):
    with open(path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        # get the first page
        page = pdf.pages[10]
        print(page)
        print('Page type: {}'.format(str(type(page))))
        text = page.extract_text()
        print("TEXT: ", text)
    
if __name__ == '__main__':
    path = 'uploaded_files/Eicher-Q1FY24-Earnings-Call-Transcript.pdf'
    path = "rag-on-steriods/financial-reports\eicher.pdf"
    text_extractor(path)