from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import gensim
import numpy as np

UTD_Class = []
MMAct_Class = []
Berkeley_Class = []

for line in open("UTD.txt","r"): #设置文件对象并读取每一行文件
    UTD_Class.append(line[:-1])  
for line in open("MMAct.txt","r"): #设置文件对象并读取每一行文件
    MMAct_Class.append(line[:-1]) 
for line in open("Berkeley.txt","r"): #设置文件对象并读取每一行文件
    Berkeley_Class.append(line[:-1]) 

glove_input_file = "glove.840B.300d.txt"
word2vec_output_file="glove.840B.300d.word2vec.txt"
(count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
print(count, '\n', dimensions)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file,binary=False)


UTD_vec=np.zeros(shape=[1,300],dtype=float) 
for i in range(0,len(UTD_Class)):
    UTD_vec_data=np.zeros((1,300),dtype=float)
    for data in UTD_Class[i].split():
        UTD_vec_data=UTD_vec_data+glove_model[data]
    #UTD_vec_data=UTD_vec_data/len(UTD_Class[i].split())   
    UTD_vec=np.vstack((UTD_vec,UTD_vec_data))
UTD_vec=UTD_vec[1:,:]
np.save('UTD_Glove.npy',UTD_vec)



MMAct_vec=np.zeros(shape=[1,300],dtype=float) 
for i in range(0,len(MMAct_Class)):
    MMAct_vec_data=np.zeros((1,300),dtype=float)
    for data in MMAct_Class[i].split():
        MMAct_vec_data=MMAct_vec_data+glove_model[data]
    #MMAct_vec_data=MMAct_vec_data/len(MMAct_Class[i].split())
    MMAct_vec=np.vstack((MMAct_vec,MMAct_vec_data))
MMAct_vec=MMAct_vec[1:,:]
np.save('MMAct_Glove.npy',MMAct_vec)


Berkeley_vec=np.zeros(shape=[1,300],dtype=float) 
for i in range(0,len(Berkeley_Class)):
    Berkeley_vec_data=np.zeros((1,300),dtype=float)
    for data in Berkeley_Class[i].split():
        Berkeley_vec_data=Berkeley_vec_data+glove_model[data]
    #Berkeley_vec_data=Berkeley_vec_data/len(Berkeley_Class[i].split())
    Berkeley_vec=np.vstack((Berkeley_vec,Berkeley_vec_data))
Berkeley_vec=Berkeley_vec[1:,:]
np.save('Berkeley_Glove.npy',Berkeley_vec)    