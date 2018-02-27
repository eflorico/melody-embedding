import subprocess
import numpy as np
import json
import scipy.io as sio

# Paths
id_file = '../results/melody-ids.csv'
embedding_file = '../results/embedding.npy'
score_file = '../datasets/match_scores.json'
tagtraum_file = '../datasets/msd_tagtraum_cd2c.cls'

# Count records
last_line = subprocess.check_output(['tail', '-1', id_file])
num_records = int(last_line.split(',')[0]) + 1
print '%d records in total' % num_records

# Load data into correct shape
embedding = np.memmap(embedding_file, dtype='float32', mode='r')
assert(embedding.shape[0] % num_records == 0)
embedding.shape = (num_records, embedding.shape[0] / num_records)

# Non-zero entries
ids_in_use = np.nonzero(np.any(embedding, axis=1))[0]
print '%d non-zero records' % len(ids_in_use)

# Load filenames whose embedding has been learned
id_md5 = {}
with open(id_file) as f:
	i = 0
	for line in f:
		id, filename = line.split(',')
		id = int(id)
		if ids_in_use[i] == id:
			md5 = filename.split('.')[0].strip()
			id_md5[id] = md5
			i += 1
			if i >= len(ids_in_use):
				break
print '%d filenames found' % len(id_md5)

# Genre data
msd_genre = {}
genres = {}
with open(tagtraum_file) as f:
	for line in f:
		if line[0] != '#':
			msd, genre = line.split("\t")
			genre = genre.strip()
			msd_genre[msd] = genre
			genres[genre] = 1
genre_idx = {}
for i, genre in enumerate(sorted(genres.keys())):
	genre_idx[genre] = i
print genre_idx

# Match scores
with open(score_file) as f:
	msd_scores = json.load(f)

md5_scores = {}
for msd, md5_with_scores in msd_scores.items():
	for md5, score in md5_with_scores.items():
		md5_scores.setdefault(md5, {})[msd] = score
print '%d md5s have matched msds' % len(md5_scores)

# Match songs and genres 
genre_probs = np.zeros((num_records, len(genres)))
for id, md5 in id_md5.items():
	for msd, score in md5_scores[md5].items():
		if msd in msd_genre:
			genre_probs[id, genre_idx[msd_genre[msd]]] += score

# File ids
md5_fids = {}
for fid, md5 in enumerate(sorted(id_md5.values())):
	md5_fids[md5] = fid
id_fids = np.zeros(num_records)
for id, md5 in id_md5.items():
	id_fids[id] = md5_fids[md5]

ids_with_genre = np.nonzero(np.any(genre_probs, axis=1))[0]

output = np.zeros((num_records, 2 + len(genres) + embedding.shape[1]))
output[:, 0] = xrange(num_records)
output[:, 1] = id_fids
output[:, 2:len(genres)+2] = genre_probs
output[:, len(genres)+2:] = embedding

output = output[np.intersect1d(ids_in_use, ids_with_genre), :]

output[:, 2:len(genres)+2] /= output[:, 2:len(genres)+2].sum(axis=1, keepdims=True)

print '%d records have genres' % output.shape[0]
print '%d files have genres' % np.unique(output[:, 1]).shape[0]

print '0: Melody ID'
print '1: File ID'
print '2-%d: Genre probabilities' % (len(genres) + 1)
print '%d-%d: Embedding' % (len(genres) + 2, output.shape[1])

np.save('data.npy', output)
#sio.savemat('embeddings.mat', {'embeddings':output})











