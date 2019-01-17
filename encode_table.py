# labels = {}
# index = 0

# f = open('./data/output.csv', 'r')
# label_output = open('./data/encode_labels', 'w')
# i = 0
# while True:
#   line = f.readline()
#   if line is None or index == 1259:
#     break
#   label = line.split(',')[-1]
#   if label not in labels:
#     labels[label] = index
#     index = index + 1
#   i = i + 1

# f.close()
# label_output.close()


encode_labels_file = open('./data/encode_labels', 'r')
labels = {}
while True:
  line = encode_labels_file.readline()
  if not line:
    break
  line = line.strip('\n')
  line = line.split(':')
  index = int(line[0])
  class_name = line[-1]
  #print(str(index) + ':' + class_name)
  labels[class_name] = index

# for key in labels:
#   print(str(key) + ':' + str(labels[key]))