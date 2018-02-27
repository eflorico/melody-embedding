# Magenta

The magenta folder contains my fork of Project Magenta. I added the following flags in addition to the flags already present in the Magenta source code:

models/melody_rnn/melody_rnn_create_dataset \
--learn_initial_state=1
Creates a csv file melody-ids.csv in the output directory that contains the mapping of record ids and filenames. The record ids are retained in the dataset for later identification.

models/melody_rnn/melody_rnn_train \
--learn_initial_state=1 \
--id_file=melody-ids.csv
Includes the initial state in training. The id file is only needed to determine the number of records.
The embedding will be output in the training dir under the name embedding.npy.

models/melody_rnn/melody_rnn_generate \
--learn_initial_state=1 \
--id_file=all_matched.recs/melody-ids.csv \
--record_a=333
--record_b=444
Uses the initial state learned for a given record in generation (if record_a is given) or the average initial state of two records (if record_b is also given).

New commands:

models/melody_rnn/melody_rnn_export_training_melody \
--sequence_example_file=... \
--record_id=##,##,## \
--config=... \
--hparams=...
Exports the specified record ids from the training set as midi files in the current directory.

Other than that, Magenta is run as usual, see the description in the Magenta repository. Build have to be run with Bazel.

# Classification

The eval folder contains the scripts preprocess.py and eval.py for classification. It requires melody-ids.csv and embedding.npy to be placed in the resuls folder and the files match_scores.json from the Lakh MIDI dataset and msd_tagtraum_cd2c.cls from the Tagtraum Genre Database to be placed in the datasets folder.


# Output

The output folder contains midi files generated when sampling using trained input states. The numbers in the folder names indicate which records were used for the initial state; two numbers indicate blending of initial states.