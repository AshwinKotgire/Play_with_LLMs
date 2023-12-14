import json
import fitz  # PyMuPDF
from openai import OpenAI
import openai
import os
client = OpenAI(api_key='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')

def pipe_to_extract_json(path_to_pdf:str,file_save_path=None)->list:
  if(file_save_path==None):
    file_save_path=path_to_pdf[:-4]+'_output.json'
  def search_multiple_keywords_in_pdf(pdf_path, keywords):
      # Open the PDF file
      pdf_document = fitz.open(pdf_path)
      print(keywords)
      # Iterate through each page in the PDF
      pages=[]
      for page_number in range(pdf_document.page_count):
          # Get the page
          page = pdf_document[page_number]

          # Check if all keywords are present on the page
          if all(keyword.lower() in page.get_text("text").lower() for keyword in keywords):
              print(f"Page {page_number + 1} contains all keywords: {', '.join(keywords)}")
              pages.append(page.get_text("text").lower())
      # Close the PDF document
      pdf_document.close()
      return pages
  The_List=[[["Provide details of greenhouse gas emissions"],'Find the quantities of green house gas emissions '],
          [['Provide details of the following disclosures related to water, in the following format'],'Find the amount of water along with with their units '],
          [['Details of total energy consumption'],'Find the details of total energy consumption '],
          [['Provide break-up of the total energy consumed'],'Provide break-up of the total energy consumed '],
          [['Provide the following details related to water discharged'],'Provide the following details related to water discharged '],
          [['Provide details related to waste management by the'],'Provide details related to waste management by the entity '],
          [['Please provide details of air emissions'],'Please provide details of air emissions '],
          [['Please provide details of total Scope 3'],'Please provide details of total Scope 3 emissions ']
          ]   
  prompts=[]
  for i in The_List:
    extract=search_multiple_keywords_in_pdf(path_to_pdf,i[0])
    # print(extract)
    e=''
    for j in extract:
      e+=j
    prompt=prompt=f'''
    You have to extract relevant data in relevant format from a given extract from a document.
    Following is the extract from the document.
    Extract:{e}
    Instruction : No need to give extra explainations.
    If there is no valid data in the extract, return '(data:invalid)'.

    {i[1]}in JSON format
    Only return these in JSON format only if the extract is valid else no need to use JSON.
    Directly return the output whithout any starting words.
    If there is no valid data in the extract, return '(data:invalid)'.
    Only provide the metrics.
    Make sure the delimeters in JSON are put properly.All values must be enclosed in double quotes must be in string format.
    Give all words in small case and put underscore between words when the values have multiple words.
    Give data for all the years mentioned in the extract.'''
    
    prompts.append(prompt)
  
  json_outputs=[]
  for p in prompts:
    stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": p}])
    result_in_json=stream.choices[0].message['content']
    json_outputs.append(json.loads(result_in_json))
  with open(file_save_path, "w") as file:
    json.dump(json_outputs, file, indent=2)  
  return json_outputs,file_save_path

def main():
  if(len(os.sys.argv)==1):
    print('Give a valid path to brsr document in pdf format')
  elif(len(os.sys.argv)==2):  
    pipe_to_extract_json(os.sys.argv[1],file_save_path=None)
  elif(len(os.sys.argv)==3):
    pipe_to_extract_json(os.sys.argv[1],file_save_path=os.sys.argv[2])
if __name__=='__main__':
  main()
