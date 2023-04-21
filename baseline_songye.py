#!/usr/bin/env python
# coding: utf-8

# In[13]:


import re
import torch
import os
import pandas as pd
from ipywidgets import widgets


# In[14]:


print(torch.__version__)
print(torch.cuda.is_available())


# In[15]:


with open('./gatsby(eng)_re.txt', 'r') as f: 
    text = f.read()


# # 1. 챕터별로 글 나누기 

# In[16]:


chapters = ['I','II','III','IV','V','VI','VII','VIII','IX']
sentence_df = {}
l = re.split(r'.*.I\n',text,1)
book_by_chapters = {}
for i in range(len(chapters)-1):
    text = l[1]
    l = re.split(r'.*.{}\n'.format(chapters[i+1]),text,1)
    sentence_df[chapters[i]] = l[0]
    sentence_df[chapters[i+1]] = l[1]
    
sentence_df


# In[18]:


#데이터프레임으로 만들어주기
sentences_series = pd.Series(sentence_df)
sentences_df = pd.DataFrame(sentences_series, columns = ['corpus'])
sentences_df


# # 2. 텍스트 전처리

# In[19]:


#텍스트 전처리해주기 
def processing_text(text):
    processed_text = re.sub('\n\n'," ",text) 
    processed_text = re.sub('\n\n-+\n\n',' ',processed_text)
    return processed_text

sentences_df['corpus'] = sentences_df['corpus'].apply(processing_text)


# # 3. max_length를 기준으로 하나의 chapter글 나누기

# In[20]:


def seperate_texts(max_length,sentences):
    current_idx = 0
    idxs = []
    current_length = 0
    while current_idx < len(sentences):
        #print("current idx is : ",current_idx)
        current_length += len(sentences[current_idx])
        if current_length > max_length:
            #print("current length is ",current_length)
            last_idx = current_idx+1
            idxs.append(last_idx)
            #print(idxs)
            current_length = 0
            continue
        current_idx +=1
    return idxs


# In[21]:


def make_df(max_length,label,sentences):
    
    #1. seperate texts 호출하기 
    idxs = seperate_texts(max_length,sentences)
    
    #2. df 만들기
    temp_array = []
    temp_text = ''
    for i in range(idxs[0]):
        temp_text += sentences[i]

    temp_array.append([label,temp_text])

    for i in range(len(idxs)-1):
        temp_text = ''
        for j in range(idxs[i], idxs[i+1]):
            temp_text += sentences[j]
        temp_array.append([label,temp_text]) 

    temp_array


    df = pd.DataFrame(temp_array,columns=['label', 'texts'])
    return df


# # 4. 모델에 넣을 수 있도록 concat 해주기
# ### 처리해주기

# In[22]:


chapters = ['I','II','III','IV','V','VI','VII','VIII','IX'] #챕터명 참고하기 

final_df = pd.DataFrame()
    
for i in range(len(chapters)):
    sentences = sentences_df['corpus'][chapters[i]].split('.')
    df_by_chapters = make_df(1024,chapters[i],sentences)
    final_df = pd.concat([final_df,df_by_chapters])
    
#최종결과
final_df


# # 5. pipeline에 넣고 돌리기

# In[10]:


from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# In[11]:


# 전체 문장 개수 len(final_df)


# In[12]:


def generate_summary(text):
    summary = summarizer(text, max_length=130, min_length=10, do_sample=False)
    return summary[0]['summary_text']

final_df['summary'] = final_df['texts'].apply(generate_summary)


# In[ ]:


final_df.to_csv('bart-large-cnn_summarization_result.csv')
final_df


# In[ ]:


get_ipython().system('pip install --upgrade diffusers transformers scipy')


# In[ ]:


from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
generator = torch.Generator("cuda").manual_seed(1024)


# In[ ]:


prompts = sentences_df['summary'].tolist()

images = []

for prompt in prompts:
    image = pipe(prompt, generator=generator).images[0]
    images.append(image)

for image in images:
    image.show()


# In[ ]:




