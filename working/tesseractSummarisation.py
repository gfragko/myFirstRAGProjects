import PyPDF2
from transformers import LlamaTokenizer, LlamaForCausalLM
from pdf2image import convert_from_path
import torch
import ollama
import pytesseract

print("Imports succesfull")



def extract_text_from_image(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for page_number, img in enumerate(images):
        print(f"Processing page {page_number + 1}/{len(images)}...")
        # Use Tesseract to extract text from the image
        page_text = pytesseract.image_to_string(img)
        # Append the text from this page to the overall text
        text += page_text + "\n"
    
    return text


def extract_text_from_pdf(pdf_path):
    """Function to read and extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"
    
    print("trying to find text...")   
    if not text.strip():
        print("Pdf is a scanned document...")
        text = extract_text_from_image(pdf_path)  
    else:
        print("The string contains visible characters.")  
        
    return text

def chunk_text_with_overlap(text, chunk_size=512, overlap_size=128):
    """Function to split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        # Move the start position to create an overlap with the previous chunk
        start += chunk_size - overlap_size
    return chunks



def summarize_pdf(pdf_path):
    """Main function to summarize the entire PDF"""
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    # print("PDF text:", pdf_text)

    # Step 2: Split text into overlapping chunks
    chunks = chunk_text_with_overlap(pdf_text, chunk_size=512, overlap_size=128)
    
    # Step 3: Summarize each chunk and combine summaries
    summary = ""
    for chunk in chunks:
        chunk_summary = summarize_chunk_with_ollama(chunk)
        # print("Chunk summary:",chunk_summary)
        summary += chunk_summary + "\n\n"

    return summary

def summarize_chunk_with_ollama(chunk):
    model_name = "llama3.1:latest"
    response = ollama.generate(model_name, prompt="You are an AI program that summarizes files for a shipping company. Summarize the following chunk of text without adding further comments (take into consideration previous chunks that you have read that overlap with this one): "+chunk)
    print(response['response'])# return response['output']  # Adjust according to the response structure
    return response['response']  # Adjust according to the response structure
    



# Example usage
pdf_path = "C:\\Users\\gfrag\\Desktop\\Workspace\\212144_I_HBL.pdf"  # Path to your PDF
pdf_summary = summarize_pdf(pdf_path)
print("Summary of image: ",pdf_summary)


with open("tesseractOut.txt", "w", encoding="utf-8") as text_file:
    text_file.write(pdf_summary)