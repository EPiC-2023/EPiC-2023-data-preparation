# EPiC processing code (for CASE dataset)

The repository contains code for processing CASE dataset [1] data (interpolated). It has been created for use in EPiC coopetition, to prepare datasets for participants.

## How to use
Script `prepare.py` contains the code that manages the process of preparing the data. It does not accept any parameters. To change the way the dataset is prepared, one can change contents of the `config.toml` file or one of the default json files in `defaults` directory. 

### Process parameters

#### config.toml
File config.toml contains settings, such as:

- `source_data_dir` - directory where the interpolated version of CASE dataset is stored
- `out_data_dir` - output directory for generated files
- `scenarios_to_generate` - list of scenarios to generate in the run of `prepare.py`
- `signals_to_include` - list of signals to inlcude in the generated data
- `random_seed` - random seed used in random processes, such as shuffling participants or video ids, or generating random folds
- `time_scale` - number of time-units per second (if time in the dataset is reported in miliseconds, then `time_scale` should be 1000, because 1s = 1000ms)
- `stimuli_labels_path` - path to the file with names of stimuli
- `default_times_dict_path` - path to the file with default parameters for time dictionary
- `all_available_signals` - all signals available in the original dataset (not used in code, just a note in case someone edited `signals_to_include` and wanted to quickly go back)

Moreover the file contains settings for processing of different scenarios, marked as `settings.scenario_<scenario num>_settings`. All time variables should be provided in seconds.
- `beginning_train_separation_time` - time between the start of a file, and start of a generated training data
- `train_time` - time of a generated training data
- `train_test_separation_time` - time between the end of training data, and beginning of test data (time separating two datasets)  
- `test_ignore_time_front` - time in the test set, where annotations (dependent variable) will not be saved, to allow participants of the challenge to utilize solutions operating on bigger-than-default time windows
- `test_time` - length of **annotated** data in the test set (the whole test set has length of `test_ignore_time_front + test_time + test_ignore_time_end`) 
- `test_ignore_time_end` - similar to `test_ignore_time_front`, but after the annotations, to the end of test set
- `kfold_num_splits` - **only scenario 2** - number of splits for k-fold cross validation (done on participants)
- `scenarios_folds_path` - **only scenarios 3 & 4** - path to .json file containing predefined splits for scenarios 3 & 4 (acoss video, and across version) 

If one of **time variables** is provided as -1, the script will assign the whole remaining time to that variable (`time of session - sum(all non-negative time variables)`). Results in different amounts of time for different stimuli.

#### defaults/default_times_dict.json

The file containing default times for the time dictionary - explanation of fields (time variables) above. Used in scenarios where training and test data come from different stimuli (folded data in scenarios 2, 3, 4). By default all values should be set to 0.

#### defaults/scenarios_folds.json

The file containing predefined folds for scenarios 3 & 4.

#### defaults/stimuli_labels.json

The file containing mapping from stimuli ids, to respective labels and arousal-valence ratings from the original dataset.

## Scripts details

The repository contains four files with python code in total.

#### prepare.py

The main script for processing the data - loads configs, initialized objects for data reading and processing, and delegates the work. Can be used as is. Processing details should be controlled via `config.toml` file.

#### data_preparation/read_and_process.py

The file containing code for `CaseDatasetReader` and `CaseDatasetProcessor`, used for reading data and processing it respectively. Reader provides methods for loading data from the hard drive and iterating it. Processor provides methods for processing data, extracting relevant time windows and saving generated data on the hard drive. 

#### data_preparation/generate_scenarios.py

The file containing the code for generating scenarios. Due to their similarity, scenarios 2, 3 & 4 utilize for the most part the same processing, thus they are handled by 1 common function.

#### simple_tests.py

The file containing simple tests for checking if generated data is OK. Tests are not sophisticated and focus mainly on ensuring that training and test sets are disjoint.

## Generated data structure

Generated dataset has following structure:

```
data directory
+----  old_new_subjects_map.json
+----  old_new_videos_map.json
+----  scenario_1
|      +----  train
|             +----  annotations
|                    +---- sub_1_vid_1.csv
|                    +---- ...
|             +----  physiological
|                    +---- sub_1_vid_1.csv
|                    +---- ...
|      +----  test
|             +---- ...
+----  scenario_2
|      +----  fold_0
|             +----  train
|                    +----  annotations
|                           +---- sub_1_vid_1.csv
|                           +---- ...
|                    +----  physiological
|                           +---- sub_1_vid_1.csv
|                           +---- ...
|              +----  test
|                     +---- ...
|      +----  ...
+----  ...

```

The structure is similar for all files, i.e. `{train, test} / {annotations, physiologicals} / <data files>`. Scenarios 2, 3, and 4 all have an additional level with number of fold, so paths look like `fold_<num> / {train, test} / {annotations, physiologicals} / <data files>`

## Literature
[1] Sharma, K., Castellini, C., van den Broek, E.L. et al. A dataset of continuous affect annotations and physiological signals for emotion analysis. Sci Data 6, 196 (2019). https://doi.org/10.1038/s41597-019-0209-0