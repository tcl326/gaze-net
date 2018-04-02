# make dataset from npz
# put this file outside data_collection folder

from scipy import misc as sci
import numpy as np
import os
import math
from shutil import copyfile

data_dir = 'data_collection/'
dst_data_dir = 'gaze_dataset/'
fps = 30

# load npz data
file = np.load(data_dir + 'annotation.npz')
name_list = file['arr_0']
target_list = file['arr_1']
start_list = file['arr_2']
end_list = file['arr_3']
print(target_list)

# make datasets
for i in range(6):
	directory = os.path.dirname(dst_data_dir + str(i) + '/')
	if not os.path.exists(directory):
		os.makedirs(directory)

num = name_list.shape[0]
valid_period = []
for i in range(num):
	print(i)
	name = name_list[i]
	target = target_list[i]
	start_count = start_list[i]
	end_count = end_list[i]
	valid_period.append([start_count, end_count])

	directory = os.path.dirname(dst_data_dir + str(target) + '/' + name)
	if not os.path.exists(directory):
		os.makedirs(directory)

	# same dataset
	add = ''
	if i != 0 and name == name_list[i-1]:
		add = '_1'
	img_dst_dir = dst_data_dir + str(target) + '/' + name + add + '/'

	# save valid images
	for j in range(start_count, end_count):
		count = str("%06d" % j)
		img_name = data_dir + name + '/' + name + count + '.jpg'
		img_dst_name =  img_dst_dir + count + '.jpg'
		# no input image
		if not os.path.exists(img_name):
			continue
		# no output dir
		if not os.path.exists(img_dst_dir):
			os.makedirs(img_dst_dir)
		copyfile(img_name, img_dst_name)

	# save valid gaze txt
	gaze_src_path = data_dir + name + '/' + name + 'testfile.txt'
	gaze_dst_path = img_dst_dir + 'gaze.txt'
	gaze_src_file = open(gaze_src_path)
	gaze_dst_file = open(gaze_dst_path, 'w')
	gaze_list_ori = gaze_src_file.readlines()
	gaze_src_file.close()
	gaze_list = gaze_list_ori[3*start_count:3*end_count]
	gaze_dst_file.writelines(gaze_list)
	gaze_dst_file.close()

	# save invalid images
	if i != num - 1 and name != name_list[i+1]:
		num_per = len(valid_period)
		start_valid = valid_period[0][0]
		end_valid = valid_period[num_per-1][1]

		for s, e in [[0, start_valid], [end_valid, len(gaze_list_ori)//3]]:
			if s >= e:
				continue
			add = ''
			if s != 0:
				add = '_1'
			img_dst_dir = dst_data_dir + str(5) + '/' + name + add + '/'
			# print(img_dst_dir)
			for j in range(s, e):
				count = str("%06d" % j)
				img_name = data_dir + name + '/' + name + count + '.jpg'
				img_dst_name =  img_dst_dir + count + '.jpg'
				# no input image
				if not os.path.exists(img_name):
					continue
				# no output dir
				if not os.path.exists(img_dst_dir):
					os.makedirs(img_dst_dir)
				copyfile(img_name, img_dst_name)
			
			# save valid gaze txt
			gaze_dst_path = img_dst_dir + 'gaze.txt'
			gaze_dst_file = open(gaze_dst_path, 'w')
			gaze_list = gaze_list_ori[3*s:3*e]
			gaze_dst_file.writelines(gaze_list)
			gaze_dst_file.close()

		valid_period = []
		print('clear')




