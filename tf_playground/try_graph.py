import tensorflow as tf

# class collection_n_element:
#     def __init__(self):

g = tf.Graph()
with g.as_default():
    a = tf.constant(10.0)
    b = tf.constant(20.0)
    c = tf.constant(30.0)

    g.add_to_collection(a, 'a')  # named by tensorflow
    g.add_to_collection(b, 'b')
    g.add_to_collection(c, 'a')
    print(g)

gg = tf.Graph()
with gg.as_default():
    a = tf.constant(100.0)
    b = tf.constant(200.0)
    c = tf.constant(300.0)

    gg.add_to_collections('a',a)  # named by user
    gg.add_to_collections('b',b)
    gg.add_to_collections('a',c)
    print(gg)

    # assert c.graph is g

# 1. Using Graph.as_default():
ggg = tf.Graph()
with ggg.as_default():
    c = tf.constant(5.0)
    print(ggg)
    assert c.graph is ggg

# 2. Constructing and making default:
with tf.Graph().as_default() as g:
    c = tf.constant(5.0)
    print(g)
    assert c.graph is g



print(ggg.as_graph_element(c))


# if __name__ == "__main__":
#     cne = collection_n_element()
