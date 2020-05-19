import tensorflow as tf
import numpy as np

t = tf.constant([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                 [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                 [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])
# tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
# tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
#                                    #   [4, 4, 4]]]
# tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                   #  [[5, 5, 5]]]

with tf.Session() as sess:
	print(t)
	print(tf.slice(t, [0, 2, 0], [3, 1, 3]).eval())

quit()

filename_queue = tf.train.string_input_producer(['/home/phantanphuc/Desktop/image/Space-Free-Download-PNG.png']) #	list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.



# def tfRGB2LAB(rgb):
# 	a_con = tf.pow((rgb + 0.055) / 1.055, 2.4)
# 	b_con = rgb / 12.92
# 	rgb_2 = tf.where(rgb.greater(0.04045), a_con, b_con);

# 	r, g, b = rgb_2.split(3, 2);

# 	x = ((r * 0.4124) + (g * 0.3576) + (b * 0.1805)) / 0.95047
# 	# y = r.mul(0.2126).add(g.mul(0.7152)).add(b.mul(0.0722)).div(1.0),
# 	# z = r.mul(0.0193).add(g.mul(0.1192)).add(b.mul(0.9505)).div(1.08883),
# 	# xyz = tf.concat([x, y, z], 2);

# 	# xyz = tf.where(
# 	# 	xyz.greater(0.008856),
# 	# 	tf.pow(xyz, 1 / 3),
# 	# 	xyz.mul(7.787).add(16 / 116)
# 	# );
# 	# x, y, z = xyz.split(3, 2);
# 	# return tf.concat([y.mul(116).sub(16), x.sub(y).mul(500), y.sub(z).mul(200)], 2)

# 	return x


def tfRGB2LAB(rgb):
	a_con = tf.pow((rgb + 0.055) / 1.055, 2.4);
	b_con = rgb / 12.92
	rgb_2 = tf.where( tf.greater(rgb, 0.04045), a_con, b_con);

	print(rgb_2)
	print(tf.split(rgb_2, (3, 2)))
	print('--------------')

	r, g, b = tf.split(rgb_2, (3, 2))

	x = ((r * 0.4124) + (g * 0.3576) + (b * 0.1805))  / 0.95047
	y = ((r * 0.2126) + (g * 0.7152) + (b * 0.0722))  / 1.0
	z = ((r * 0.0193) + (g * 0.1192) + (b * 0.9505))  / 1.08883
	
	xyz = tf.concat([x, y, z], 2);

	xyz = tf.where(tf.greater(xyz, 0.008856), tf.pow(xyz, 1 / 3), (xyz * 7.787) + (16 / 116))
	
	x, y, z = tf.split(xyz, (3, 2))
	
	# x, y, z = xyz.split(3, 2);
	return tf.concat([y.mul(116).sub(16), x.sub(y).mul(500), y.sub(z).mul(200)], 2)



init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)

	# Start populating the filename queue.

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(1): #length of your filename list
		image = my_img #here is your image Tensor :) 



		img_hsv = tf.image.rgb_to_hsv(tf.image.convert_image_dtype(my_img, dtype=tf.float32))
		image_hsv = img_hsv.eval() #here is your image Tensor :) 

		temp1 = tf.image.convert_image_dtype(my_img, dtype=tf.float32).eval() #here is your image Tensor :) 



	# print(image.shape)
	# print(image)

	# print(my_img.eval())
	print(tfRGB2LAB(tf.image.convert_image_dtype(my_img, dtype=tf.float32)))

	# temp3 = tf.image.resize_images(image, size=(480, 720))
	# temp3 = tf.image.convert_image_dtype(temp3, dtype=tf.float32, saturate=False)

	# print(temp3.eval())

	# print(temp1)

	print('---------------------------------------')
	print('---------------------------------------')
	print('---------------------------------------')

	# print(temp1)
	# print('---------------------------------------')
	# print(image_hsv)
	# print('---------------------------------------')
	# print(image_hsv)

	# Image.fromarray(np.asarray(image)).show()

	coord.request_stop()
	coord.join(threads)

	quit()
















quit()


import tensorflow as tf
import numpy as np
from reader import Reader

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# sess = tf.Session(config=run_config)
sess = tf.Session(config=run_config)

nbins = 5
value_range = [0.0, 5.0]
new_values = [[-1.0, 1, 1, 1, 1, 0.0, 1.5, 2.0, 5.0, 15]]
# imtesr = [[[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]]]
imtesr = [[[[1,33,3], [1,33,6]], [[1,33,1], [11,33,1]]], [[[2,22,1], [31,32,33]], [[2,42,43], [2,52,53]]]\
, [[[21,22,1], [31,32,33]], [[41,42,43], [51,52,53]]], [[[21,22,1], [31,32,33]], [[41,42,43], [51,52,53]]]]
# imtesr = [[[[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]]], [[[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]]]]

# with sess as sess:

# 	test_tesor = tf.convert_to_tensor(imtesr)
# 	print(test_tesor)

# 	with tf.variable_scope('color_hist_producer') as scope:
# 		bin_size = 1
# 		hist_entries = []

# 		img_r, img_g, img_b = tf.split(test_tesor, 3, axis=3)
# 		# img_r, img_g, img_b = tf.split(2, 2, 2, 3, test_tesor, axis=1)

# 		for img_chan in [img_r, img_g, img_b]:
# 			for idx, i in enumerate(np.arange(0, 100, bin_size)):
# 				gt = tf.greater(img_chan, i)
# 				leq = tf.less_equal(img_chan, i + bin_size)

				
# 				node = tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32), axis=(1,2,3))

# 				hist_entries.append(node)
# 				# hist_entries.append(tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32), axis=1, keepdims=True))


# 		hist = tf.stack(hist_entries)

# 		hist = tf.transpose(hist)

# 		print(hist.eval())
# 		print(hist)

imtesr = [[[[1,1,1], [1,33,1]], [[11,33,1], [11,33,1]]], [[[2,22,1], [31,32,33]], [[2,42,43], [2,52,53]]]\
, [[[21,22,1], [31,32,33]], [[41,42,43], [51,52,53]]], [[[21,2,3], [2,2,33]], [[41,1,43], [51,52,53]]]]


with sess as sess:

	test_tesor = tf.convert_to_tensor(imtesr)

	with tf.variable_scope('color_hist_producer') as scope:
		bin_size = 1
		hist_entries = []

		for idx, i in enumerate(np.arange(0, 100, bin_size)):
			gt = tf.greater(test_tesor, i)
			leq = tf.less_equal(test_tesor, i + bin_size)

			node = tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32), axis=(1, 2))
			hist_entries.append(node)


		hist = tf.stack(hist_entries)
		hist = tf.transpose(hist, perm=[1, 2, 0])
		hist = tf.reshape(hist, shape=(4, 300))
		# (2, 256, 512, 3)
		# hist = tf.nn.l2_normalize(hist, 0)

		hist = tf.tile(hist, [1, 6])
		hist = tf.reshape(hist, shape=(4, 2, 3, 300))

		hist = hist / 4


		# out = tf.concat([hist, hist], axis=3)

		# hist = tf.math.normalize(hist, ord='euclidean', axis=2)

		diff = tf.reduce_sum(hist - hist)

		print(diff.eval())
		print(diff)
		# print(out)


quit()
 