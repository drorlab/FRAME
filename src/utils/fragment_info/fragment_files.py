def read_fragment_names_from_file(filename):
	with open(filename, 'r') as f:
		result = [line.rstrip() for line in f]
	return result


