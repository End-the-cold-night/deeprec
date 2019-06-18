# fm
import tensorflow as tf
def fm(embeddings):
    # input: embeddings should be batch * filed * embedding_dim,
    # output: fm, 
    # consider if summed_features_emb and squared_features_emb can be returned !!!!!
    
    summed_features_emb = tf.reduce_sum(embeddings,1) # batch * k
    summed_features_emb_square = tf.square(summed_features_emb) # batch * K
    
    
    squared_features_emb = tf.square(embeddings)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # batch * K
    
    
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square,squared_sum_features_emb) #batch * K
    
    return  y_second_order


 
