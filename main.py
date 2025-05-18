import numpy as np  

word_embeddings = [] 
position_embeddings = [] 
sentence = "the dog sat"
words = sentence.split()  
final_embeddings = []

for i in range(len(words)):
    vector_4d = np.random.rand(4)  
    
    pos_vector = np.array([np.sin(i), np.cos(i), np.sin(i), np.cos(i)])

    combined_vectors = vector_4d + pos_vector
    
    word_embeddings.append(word_embeddings)
    final_embeddings.append(combined_vectors)


matrix = np.array(final_embeddings)

print(matrix)
