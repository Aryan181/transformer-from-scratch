import numpy as np  

embeddings = []  
sentence = "the dog sat"
words = sentence.split()  

for word in words:
    vector_4d = np.random.rand(4)  
    embeddings.append(vector_4d)   


matrix = np.array(embeddings)

print(matrix)
