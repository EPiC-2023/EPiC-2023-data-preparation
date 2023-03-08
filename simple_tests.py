import os
import hashlib
import pandas as pd


def compute_file_hash(file_path):
    "Compute hash of a file"
    # open file
    with open(file_path, "rb") as f:
        # make hash object
        file_hash = hashlib.blake2b()
        # update hash
        while chunk := f.read(8192):
            file_hash.update(chunk)
    # return in hex form
    return file_hash.hexdigest()


def names_differences(fold_dir):
    "Check similarities and differences between names in train and test sets"
    train_annotations_names = set(os.listdir(os.path.join(fold_dir, 'train', 'annotations')))
    train_physiology_names = set(os.listdir(os.path.join(fold_dir, 'train', 'physiology')))
    test_annotations_names = set(os.listdir(os.path.join(fold_dir, 'test', 'annotations')))
    test_physiology_names = set(os.listdir(os.path.join(fold_dir, 'test', 'physiology')))
    return train_annotations_names.isdisjoint(test_annotations_names), train_annotations_names == test_annotations_names, ((train_annotations_names == train_physiology_names), (test_annotations_names == test_physiology_names))


def dir_hash_train_test_differences(fold_dir, data_type):
    "Check similarities and differences between hashes of files from training and test set"
    # storage for results
    train_hashes = set()
    test_hashes = set()
    train_paths = set()
    test_paths = set()
    # iterate train directory
    train_annot_path = os.path.join(fold_dir, 'train', data_type)
    for train_count, f_name in enumerate(os.listdir(train_annot_path), 1):
        # get path and compute hash of file
        f_path = os.path.join(train_annot_path, f_name)
        hash = compute_file_hash(f_path)
        # if hash and file already seen - duplicate
        # we can have similar hashes for participants that did not annotate sessions - false positives
        # that's why we check file names, but only for annotations
        if hash in train_hashes:
            if data_type == 'annotations':
                assert f_path not in train_paths, "Duplicated file"
            else:
                assert False, "Duplicated"
        # add to memory
        train_hashes.add(hash)
        train_paths.add(f_path)
    # iterate test directory
    test_annot_path = os.path.join(fold_dir, 'test', data_type)
    for test_count, f_name in enumerate(os.listdir(test_annot_path), 1):
        # everything same as above
        f_path = os.path.join(test_annot_path, f_name)
        hash = compute_file_hash(f_path)
        if hash in test_hashes:
            if data_type == 'annotations':
                assert f_path not in test_paths, "Duplicated file"
            else:
                assert False, "Duplicated"
        test_hashes.add(hash)
        test_paths.add(f_path)
    return train_hashes.isdisjoint(test_hashes), train_hashes == test_hashes, ((train_count, len(train_hashes)), (test_count, len(test_hashes)))


def scenario_folds_test_sets_hash_differences(scenario_dir):
    "Check file names and hashes between test folds for a scenario"
    # memory
    test_hashes = set()
    test_paths = set()
    # for each fold directory
    for fold_dir in os.listdir(os.path.join(scenario_dir)):
        # for each data type
        for data_type in 'annotations', 'physiology':
            test_path = os.path.join(scenario_dir, fold_dir, 'test', data_type)
            f_paths = [os.path.join(test_path, f_name) for f_name in os.listdir(test_path)]
            for f_path in f_paths:
                # compute hash
                curr_hash = compute_file_hash(f_path)
                # if hash and file already seen - duplicate
                # we can have similar hashes for participants that did not annotate sessions - false positives
                # that's why we check file names, but only for annotations
                if curr_hash in test_hashes:
                    if data_type == 'annotations':
                        assert f_path not in test_paths, "Test folds are not disjoint"
                    else:
                        assert False, "Test folds are not disjoint"
                # save hash and path
                test_hashes.add(curr_hash)
                test_paths.add(f_path)
    return True


def examine_data_files(path):
    "Examine csv files"
    def _check_dataframes(annot_df, physio_df):
        # check if data is not too short
        assert (len(annot_df) > 1) and (len(physio_df) > 1), "Too short files"
        # check if physiology starts from 0 time
        assert physio_df['time'].iloc[0] == 0, "Wrong start time"
        # check if annotations always have corresponding physiology
        assert all(annot_df['time'] >= physio_df['time'].iloc[0]) and all(annot_df['time'] <= physio_df['time'].iloc[-1]), "Annotations corrupt index"
    annotations_dir = os.path.join(path, 'annotations')
    physiology_dir = os.path.join(path, 'physiology')
    annotations_files = sorted(os.listdir(annotations_dir))
    for f_name in annotations_files:
        # load data
        annot_df = pd.read_csv(os.path.join(annotations_dir, f_name))
        physio_df = pd.read_csv(os.path.join(physiology_dir, f_name))
        # examine data
        _check_dataframes(annot_df, physio_df)
    return True


def test_scenario_1(scenario_1_dir):
    "Perform simple tests on scenario 1 data"
    # check differences in names
    _, is_traintest_same, (is_train_same, is_test_same) = names_differences(scenario_1_dir)
    # ok if names are the same in train and test set, and are the same in physiology and annotations
    # else return false
    if not all((is_traintest_same, is_train_same, is_test_same)):
        return False
    are_annotations_disjoint, are_annotations_same, _ = dir_hash_train_test_differences(scenario_1_dir, 'annotations')
    is_physiology_disjoint, is_physiology_same, _ = dir_hash_train_test_differences(scenario_1_dir, 'physiology')
    # ok if all hashes are different in train and test set
    # else return false
    if any((are_annotations_same, is_physiology_same)) or not all((are_annotations_disjoint, is_physiology_disjoint)):
        return False
    # examine data files
    examine_data_files(os.path.join(scenario_1_dir, 'train'))
    examine_data_files(os.path.join(scenario_1_dir, 'test'))
    # if everything ok, return True
    return True


def test_scenarios_234(scenario_dir):
    "Perform simple tests on scenarios with folds"
    # check names and hashes betweem all test folds in the scenario
    scenario_folds_test_sets_hash_differences(scenario_dir)
    # for each fold
    for fold_dir in os.listdir(scenario_dir):
        # generate path
        fold_path = os.path.join(scenario_dir, fold_dir)
        is_disjoint, is_traintest_same, (is_train_same, is_test_same) = names_differences(fold_path)
        # ok if names are different in train and test set, and are the same in physiology and annotations
        # else return false
        if not (is_disjoint and (not is_traintest_same) and all((is_train_same, is_test_same))):
            return False
        are_annotations_disjoint, are_annotations_same, _ = dir_hash_train_test_differences(fold_path, 'annotations')
        is_physiology_disjoint, is_physiology_same, _ = dir_hash_train_test_differences(fold_path, 'physiology')
        # ok if all hashes are different in train and test set
        # else return false
        if any((are_annotations_same, is_physiology_same)) or not all((are_annotations_disjoint, is_physiology_disjoint)):
            return False
        examine_data_files(os.path.join(fold_path, 'train'))
        examine_data_files(os.path.join(fold_path, 'test'))
    return True


if __name__ == '__main__':
    # scenario 1
    print('Testing scenario 1')
    scenario_1_dir = 'data/scenario_1'
    res = test_scenario_1(scenario_1_dir)
    print("OK" if res else "NOK")
    # scenarios 2, 3, 4
    for scenario in [2,3,4]:
        print(f'Testing scenario {scenario}')
        scenario_dir = f'data/scenario_{scenario}'
        res = test_scenarios_234(scenario_dir)
        print("OK" if res else "NOK")
