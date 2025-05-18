import numpy as np  

word_embeddings = [] 
position_embeddings = [] 
sentence = "the dog sat"
words = sentence.split()  
final_embeddings = []

queries = []
keys = []
values = []


for i in range(len(words)):
    vector_4d = np.random.rand(4)  
    
    pos_vector = np.array([np.sin(i), np.cos(i), np.sin(i), np.cos(i)])

    combined_vectors = vector_4d + pos_vector

    word_embeddings.append(vector_4d)
    final_embeddings.append(combined_vectors)


matrix = np.array(final_embeddings)


W_Q = np.random.rand(4, 4)
W_K = np.random.rand(4, 4)
W_V = np.random.rand(4, 4)


for embedding in final_embeddings: 
    queries.append(embedding @ W_Q)
    keys.append(embedding @ W_K)
    values.append(embedding @ W_V)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()


attention_outputs = []

for query in queries:

    scores = []

    for key in keys:
        dot = np.dot(query, key)
        scaled = dot / 2.0
        scores.append(scaled)

    weights = softmax(scores)

    final_vector = final_vector = np.sum([w * v for w, v in zip(weights, values)], axis=0)
    attention_outputs.append(final_vector)

print(np.array(attention_outputs))