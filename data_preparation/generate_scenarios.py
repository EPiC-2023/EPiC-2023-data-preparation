import os
from sklearn.model_selection import KFold
import json
from tqdm import tqdm
from pprint import pprint


def folds_loader(folds_path, scenario):
    "Load folds data for scenario from .json file in folds_path"
    # load json
    with open(folds_path, "r") as fp:
        folds_load = json.load(fp)[f"scenario_{scenario}"]
    # prepare output dict
    folds = dict()
    # for each fold
    for fold_num, fold_data in folds_load.items():
        # transform array (list) to set
        # we cannot store sets in .json files
        data_dict = {"train": set(fold_data["train"]), "test": set(fold_data["test"])}
        # save fold info
        folds.setdefault(fold_num, data_dict)
    # return folds data
    return folds


def _print_info(scenario, times_dict):
    "Print stage info"
    print("=" * 50)
    print("Prepare scenario", scenario)
    print("Scenario times dict")
    pprint(times_dict)


def generate_scenario_1(config_dict, reader, processor, out_data_dir):
    # set scenario
    scenario = 1
    # setup output directories
    out_annotations_dir = os.path.join(
        out_data_dir, f"scenario_{scenario}"
    )
    out_physiology_dir = os.path.join(
        out_data_dir, f"scenario_{scenario}"
    )
    # load times_dict with lengths of data segments
    times_dict = config_dict["settings"][f"scenario_{scenario}_settings"]
    # print generation info
    _print_info(f"scenario_{scenario}", times_dict)
    print("Progress [subjects]")
    # iterate over subjects data
    for (subject_id, (subject_annotations, subject_physiology),) in tqdm(
        reader.iterate_subjects_data()
    ):
        # for each video
        for video in reader.videos:
            # get data for the video
            video_annotations, video_physiology = processor.extract_video_data(
                subject_data=(subject_annotations, subject_physiology), video=video
            )
            # compute intervals for data splitting
            time_intervals = processor.compute_intervals(
                video_annotations["time"].iloc[0],
                video_annotations["time"].iloc[-1],
                times_dict,
            )
            # train annotations and physiology extraction
            train_annotations, train_physiology = processor.extract_data_for_intervals(
                video_annotations, video_physiology, time_intervals, "train"
            )
            # save both annotations and physiology
            processor.save_data(
                annotations=train_annotations,
                out_annotations_dir=out_annotations_dir,
                physiology=train_physiology,
                out_physiology_dir=out_physiology_dir,
                subject_id=subject_id,
                video=video,
                set_type="train",
                reset_time=True,
                reset_time_amount=train_physiology["time"].iloc[0],
            )
            # test annotations and physiology extraction
            test_annotations, test_physiology = processor.extract_data_for_intervals(
                video_annotations, video_physiology, time_intervals, "test"
            )
            processor.save_data(
                annotations=test_annotations,
                out_annotations_dir=out_annotations_dir,
                physiology=test_physiology,
                out_physiology_dir=out_physiology_dir,
                subject_id=subject_id,
                video=video,
                set_type="test",
                reset_time=True,
                reset_time_amount=test_physiology["time"].iloc[0],
            )


def generate_scenario_234(
    config_dict,
    reader,
    processor,
    out_data_dir,
    scenario,
    kfold_random_seed=None,
    save_physiology=True,
):
    def _prepare_scenario_2_folds(scenario_settings):
        # prepare folds for scenario 2 - based on participants
        train_tuple = tuple("train" for _ in reader.videos)
        test_tuple = tuple("test" for _ in reader.videos)
        # use kfold split
        kfold = KFold(
            n_splits=scenario_settings["kfold_num_splits"],
            shuffle=True,
            random_state=kfold_random_seed,
        )
        # prepare dict for folds based on subjects
        # each subject has a list of folds data
        # folds data, where positions correspond to movie_id, and values are from {train, test} set
        subjects_folds = {subject_id: list() for subject_id in reader.subjects}
        # split subjects into folds
        for train_ids, test_ids in kfold.split(reader.subjects):
            # for each index, save respective subject id (kfold returns ids of the array, that starts from 0)
            train_subjects_ids = list(
                map(lambda s_id: reader.subjects[s_id], train_ids)
            )
            # same for test subjects
            test_subjects_ids = list(map(lambda s_id: reader.subjects[s_id], test_ids))
            # for each subject in the fold mark all videos as either train or test
            for subject_id in train_subjects_ids:
                subjects_folds[subject_id].append(train_tuple)
            for subject_id in test_subjects_ids:
                subjects_folds[subject_id].append(test_tuple)
        # return folds data
        return subjects_folds

    def _prepare_scenario_34_folds(folds):
        # prepare dict for folds based on subjects
        subjects_folds = {subject_id: list() for subject_id in reader.subjects}
        # for each fold
        for _, fold_data in folds.items():
            # take train videos
            train_videos = fold_data["train"]
            # make mapping: video -> {train, test}
            video_traintest_map = {
                video_num: "train" if video_labels["label"] in train_videos else "test"
                for video_num, video_labels in reader.stimuli_labels_map.items()
            }
            # make tuple with videos marked as either train or test
            video_sets = tuple(video_traintest_map[video] for video in reader.videos)
            # save the tuple in each subject's info - every subject has the same tuple
            # made for compatibility with scenario 2 and faster processing (subject-based)
            for subject_id in reader.subjects:
                subjects_folds[subject_id].append(video_sets)
        # return folds data
        return subjects_folds

    # load settings for the scenario
    scenario_settings = config_dict["settings"][f"scenario_{scenario}_settings"]
    # prepare output directories
    out_annotations_dir = os.path.join(
        out_data_dir, f"scenario_{scenario}"
    )
    out_physiology_dir = os.path.join(
        out_data_dir, f"scenario_{scenario}"
    )
    # load times dictionary with 0s everywhere for later use
    with open(config_dict["settings"]["default_times_dict_path"], "r") as fp:
        default_times_dict = json.load(fp)
    # prepare dictionary with times for training
    train_times_dict = {
        **default_times_dict,
        "beginning_train_separation": scenario_settings["beginning_train_separation"],
        "train_time": scenario_settings["train_time"],
        "train_test_separation_time": scenario_settings["train_test_separation_time"],
    }
    # prepare dictionary with times for testing
    test_times_dict = {
        **default_times_dict,
        "train_test_separation_time": scenario_settings["train_test_separation_time"],
        "test_ignore_time_front": scenario_settings["test_ignore_time_front"],
        "test_time": scenario_settings["test_time"],
        "test_ignore_time_end": scenario_settings["test_ignore_time_end"],
    }
    # if no random seed provided take it from the config
    if not kfold_random_seed:
        kfold_random_seed = config_dict["settings"]["random_seed"]
    # set subjects_folds to None and load it based on the scenario
    subjects_folds = None
    if scenario == 2:
        subjects_folds = _prepare_scenario_2_folds(scenario_settings)
    elif scenario == 3 or scenario == 4:
        folds = folds_loader(
            scenario_settings["scenarios_folds_path"], scenario,
        )
        assert folds is not None, "Folds required for scenarios 3 and 4"
        subjects_folds = _prepare_scenario_34_folds(folds)
    assert subjects_folds is not None, "Wrong scenario"
    # print information
    _print_info(scenario, {"train": train_times_dict, "test": test_times_dict})
    print("Progress [subjects]")
    # iterate over data of all subjects
    # we iterate over sujects first, so we open data files only once
    # it speeds up the whole process a bit
    for (subject_id, (subject_annotations, subject_physiology),) in tqdm(
        reader.iterate_subjects_data()
    ):
        # get data folds for current subject
        subject_folds = subjects_folds.get(subject_id)
        # for each video
        for video in reader.videos:
            # get physiology and annotations for the video
            video_annotations, video_physiology = processor.extract_video_data(
                subject_data=(subject_annotations, subject_physiology), video=video
            )
            # for each fild in subject folds
            for fold_idx, videos_buckets in enumerate(subject_folds):
                set_type = videos_buckets[video - 1]
                # if set_type == "test":
                    # if processor.memory_dict[subject_id][video][set_type]:
                    #     continue
                    # for now we have whole subjects, but it can be easily modified using times_dict and compute_time_index method.
                    # Then, it would be best to save test physiology separately
                times_dict = test_times_dict if set_type == 'test' else train_times_dict
                time_intervals = processor.compute_intervals(
                    video_annotations["time"].iloc[0],
                    video_annotations["time"].iloc[-1],
                    times_dict,
                    allow_common_train_test_sample=True,
                )
                # extract data based on the provided intervals
                (
                    annotations,
                    physiology,
                ) = processor.extract_data_for_intervals(
                    video_annotations, video_physiology, time_intervals, set_type
                )
                # save data
                processor.save_data(
                    annotations=annotations,
                    out_annotations_dir=out_annotations_dir,
                    physiology=physiology if save_physiology else None,
                    out_physiology_dir=out_physiology_dir
                    if save_physiology
                    else None,
                    subject_id=subject_id,
                    video=video,
                    set_type=set_type,
                    fold_idx=fold_idx,
                    reset_time=True,
                    reset_time_amount=physiology["time"].iloc[0],
                )
                    # got to next fold
                #     continue
                # # get train times and timestamps for supplied intervals
                # time_intervals = processor.compute_intervals(
                #     video_annotations["time"].iloc[0],
                #     video_annotations["time"].iloc[-1],
                #     train_times_dict,
                #     allow_common_train_test_sample=True,
                # )
                # # extract data based on the provided intervals
                # (
                #     train_annotations,
                #     train_physiology,
                # ) = processor.extract_data_for_intervals(
                #     video_annotations, video_physiology, time_intervals, set_type
                # )
                # # save data
                # processor.save_data(
                #     annotations=train_annotations,
                #     out_annotations_dir=out_annotations_dir,
                #     physiology=train_physiology if save_physiology else None,
                #     out_physiology_dir=out_physiology_dir if save_physiology else None,
                #     subject_id=subject_id,
                #     video=video,
                #     set_type="train",
                #     fold_idx=fold_idx,
                # )
