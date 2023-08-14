# %%writefile Cust_chain.py

from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import TokenTextSplitter

from IPython.display import HTML

from langchain.document_loaders import OnlinePDFLoader


from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
import transformers
import torch

class Cust_Chain_obj():
  def __init__(self,model,tokenizer,FAISS_obj,sys_prompt='',embeddings = HuggingFaceEmbeddings(),text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50)):
    self.FAISS_obj=FAISS_obj
    self.model=model
    self.tokenizer=tokenizer
    self.sys_prompt=sys_prompt
    self.embedding_obj=embeddings
    self.text_splitter_obj=text_splitter
    self.doc=[]
    self.prompt_template=self.create_prompt_template()
  def create_prompt_template(self):
    prompt="""<s>[INST] """+self.sys_prompt+"""
        Context:{context}
        Qustion:{question}
        [/INST]
        """
    prompt_t=PromptTemplate(input_variables=['context','question'],template=prompt)
    return prompt_t
  def get_device(self):
    if torch.cuda.is_available():
      print('cuda')
      device = torch.device("cuda")  # If GPU is available, use it.
    else:
      device = torch.device("cpu")   # If GPU is not available, use the CPU.
    return device  


  def load_pdf_doc(self,doc_path):
    loader=PyPDFLoader(doc_path)
    data=loader.load()
    self.doc=data
    return self.doc

  def set_embedding_obj(self,new_embedding_obj):
    self.embedding_obj=new_embedding_obj

  def set_sys_prompt(self,new_sys_prompt):
    self.sys_prompt=f"<<SYS>>{new_sys_prompt}<</SYS>>"
    self.prompt_template=self.create_prompt_template()

  def set_new_model(self,new_model):
    self.model=new_model

  def populate_vector_store(self,document=None,embedding_obj=None,text_splitter=None):
    if(document==None):
      document=self.doc
    if embedding_obj is None:
      embedding_obj=self.embedding_obj
    if (text_splitter is None):
      text_splitter=self.text_splitter_obj
    docs=text_splitter.split_documents(document)
    self.FAISS_obj=FAISS.from_documents(docs,embedding_obj)

  def retrieve_documents(self,query,k=2):
    contexts=self.FAISS_obj.similarity_search(query,k=k)
    return contexts

  def retrieve_contexts(self,query,k=2,use_contexts=False,cust_context='')  :
    context = "No context available ,answer on your own.\n"
    meta_datas=[]
    if(use_contexts==True):
      contexts=self.retrieve_documents(query,k)
      cc=''
      for c in contexts:
        cc+=c.page_content
        meta_datas.append(c.metadata)
        cc+='\n'
      if(len(cc)!=0  ):
        context=cc
      context+=cust_context
    return context,meta_datas

  def run(self,query,k=2,use_contexts_from_doc=False,cust_context=''):
    context,metadata=self.retrieve_contexts(query,k,use_contexts_from_doc,cust_context)
    prompt=self.prompt_template.format(context=context,question=query)
    inputs = self.tokenizer(prompt, return_tensors="pt",padding =True).to(self.get_device())
    generate_ids = self.model.generate(inputs.input_ids, max_length=4000,top_k=1,top_p=0.5,temperature=0.01)
    p=generate_ids[0][inputs['input_ids'].shape[1]:]
    output=self.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return metadata,output
