from data_preparation.read_and_process import CaseDatasetReader, CaseDatasetProcessor
from data_preparation.generate_scenarios import (
    generate_scenario_1,
    generate_scenario_234
)
import os
import toml


# load configs
with open("config.toml", "r") as fp:
    config_read = fp.read()
    config_dict = toml.loads(config_read)

source_data_dir = config_dict["env_paths"]["source_data_dir"]
source_annotations_dir = os.path.join(source_data_dir, "annotations")
source_physiology_dir = os.path.join(source_data_dir, "physiological")

out_data_dir = config_dict["env_paths"]["out_data_dir"]

scenarios_to_prepare = set(config_dict["settings"]["scenarios_to_generate"])
signals_to_include = config_dict["settings"]["signals_to_include"]

# make instances of objects for loading and processing data
reader = CaseDatasetReader(source_annotations_dir, source_physiology_dir)
processor = CaseDatasetProcessor(reader)

random_seed = int(config_dict["settings"]['random_seed'])

if __name__ == "__main__":
    if 1 in scenarios_to_prepare:
        # scenario 1
        generate_scenario_1(config_dict, reader, processor, out_data_dir)
    if 2 in scenarios_to_prepare:
        # scenario 2
        generate_scenario_234(
            config_dict,
            reader,
            processor,
            out_data_dir,
            scenario=2,
            kfold_random_seed=random_seed
        )
    if 3 in scenarios_to_prepare:
        # scenario 3
        generate_scenario_234(
            config_dict,
            reader,
            processor,
            out_data_dir,
            scenario=3
        )
    if 4 in scenarios_to_prepare:
        # scenario 4
        generate_scenario_234(
            config_dict,
            reader,
            processor,
            out_data_dir,
            scenario=4
        )
    # save metadata
    processor.save_metadata(out_data_dir)
