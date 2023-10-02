from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
import transformers
import torch

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL,TOKENIZER=None,None
def Model_tokenizer(name:str,token):
    from huggingface_hub import login
    model,tokenizer,comment=None,None,None
    try:
        login(token)
        comment='Login successful to hugging face hub'
        model_name = name

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        return model,tokenizer,comment
    except ValueError:
        return None,None,"ValueError: Invalid token passed!"
    except:
        return None,None,"Something else except 'Invalid token' went wrong!"
    


def check_is_admin(name:str,passw:str):
    if(name=='Administrator'):
        return True
    else:return False
def normal_auth(name,passw):
    return True

async def answer_the_question(question:str,max_length=2000):
    inputs = TOKENIZER(question, return_tensors="pt",padding =True)
    generate_ids = MODEL.generate(inputs.input_ids, max_length=max_length)
    p=generate_ids[0][inputs['input_ids'].shape[1]:]
    output=TOKENIZER.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output

app = FastAPI()


class Item(BaseModel):
    name: str
    password:str
    hf_token:Union[None,str]
    # is_offer: Union[bool, None] = None

class Prompt(BaseModel):
    userid:str
    prompt:str
@app.get("/")
def read_root():
    return {"Hello": "World"}

#---------------------------------Socketio-------------------------------------------------------------
@app.post('/authenticate')
async def authenticate(input:Item):
    global MODEL_NAME,MODEL,TOKENIZER
    print(input.name,'    ',input.password,input.hf_token)
    if(check_is_admin(input.name,input.password)):
        MODEL,TOKENIZER,comment=Model_tokenizer(MODEL_NAME,input.hf_token)
        if(MODEL==None or TOKENIZER==None):
            return {'auth':False,'comment':comment}
        else:
            return {'auth':True,'comment':comment}
    elif(normal_auth(input.name,input.password)):
        return {'auth':True,'comment':'Login successful'}
    else:
        return {'auth':False,'comment':'Login unsuccessful'}
    




    return {'auth':True}
@app.post('/answer_prompt')
async def answer(input:Prompt):
    print(input.prompt)
    answer= await answer_the_question(input.prompt)
    return {'answer':answer}


if __name__=='__main__':
    # th1=threading.Thread(None,handle_firestore)
    # th1.start()
    uvicorn.run("main:app", port=8000, log_level="info",host='0.0.0.0')
    # th1.join()

