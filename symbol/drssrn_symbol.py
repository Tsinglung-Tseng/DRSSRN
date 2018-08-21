if __name__ == '__main__':
    g = DataGen(file)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a,b,c = g.next_batch

        d = DownSampler(a, 8, g.batch_size)

    with tf.Session() as sess:
        img_down_sample_x2 = sess.run(d())

    # plt.imshow(img_down_sample_x2.reshape(d.output_size))
    # plt.colorbar()
    # plt.show()

    # img = img_down_sample_x2[1]
    # plt.imshow(img.reshape(d.output_size[1:3]))
    # plt.colorbar()
    # plt.show()
