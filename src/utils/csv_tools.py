import csv
import pickle

def write_csv(file_name, columns, array_dict):
	if len(columns)==0:
		columns = array_dict[0].keys()
	with open(file_name, 'w') as f: 
		writer = csv.DictWriter(f, fieldnames = columns)
		writer.writeheader()
		for r in array_dict:
			writer.writerow(r)


def read_csv(file_path):
    with open(file_path, "r") as f:
        a = list(csv.DictReader(f))
    return a

def pickle_read(path):
	with open(path, 'rb') as handle:
		data = pickle.load(handle)
	return data

def pickle_write(path, data):
	with open(path, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
