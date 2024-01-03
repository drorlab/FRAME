## Running Frame
General code for running FRAME. 

Running FRAME for a specific project should be run by importing 
these functions into the appropriate file. Please don't add configs and data structures
to these files. 

### Running FRAME 
Run frame using code like so: 
```
from src.frag_adder.run_FRAME import run_FRAME
run_name = '2023_05_22_pairs_ML_2'

# Create a config object, usually by copying and modifying existing
default_config_ML5 = copy.deepcopy(default_config_ML4)
default_config_ML5['output_root_folder'] = '/scratch/groups/rondror/lxpowers/temp_output/FRAME_temp_outputs'

#Create a dataset object that implements interface frag_adder/I_FRAME_ProviderMixin
dataset = Pair_dataset()

#Get the ids you want to run in this batch
ids = dataset.get_benchmark_pairs_smaller()
ids = ids[index_start:index_start+number_items]

#Start running Frame 
run_FRAME(dataset, config, ids, run_name)
```


### Using the CLI 
Use `CLI.py` to access the FRAME CLI from your own file, which will wrap your commands to submit batches of sherlock jobs.
You need to provide two function, `run_tasks` and `get_ids`. 

```
from src.frag_adder.CLI import FRAME_CLI
from src.frag_adder.run_FRAME import run_FRAME

def run_tasks(index_start, number_items):
    run_name = '2023_05_22_pairs_ML_2'
    default_config_ML5 = copy.deepcopy(default_config_ML4)
    default_config_ML5['output_root_folder'] = '/scratch/groups/rondror/lxpowers/temp_output/FRAME_temp_outputs'
    dataset = Pair_dataset()
    ids = dataset.get_benchmark_pairs_smaller()
    ids = ids[index_start:index_start+number_items]
    run_FRAME(dataset, config, ids, run_name)

def get_ids():
    dataset = Pair_dataset()
    ids = dataset.get_benchmark_pairs_smaller()
    return ids



if __name__ == '__main__':
    output_root = '/scratch/groups/rondror/lxpowers/temp_output/FRAME_temp_outputs/2023_05_22_pairs_ML_2/'
    
    # This is module path to this file
    file = 'src.users.koodli.pair_run_frame'
    FRAME_CLI(run_tasks, get_ids, output_root, file)
```

Now, you can call your file from the command line, using frame CLI options. 
```
$SCHRODINGER/run python3 -m src.users.koodli.pair_run_frame submit_jobs --dry_run
```
You have a lot of options for controlling jobs: 
```
[--start_index START_INDEX] [--total TOTAL] [--number_per_job NUMBER_PER_JOB]
                         [--time TIME] [--partition PARTITION] [--dry_run] [--submit_all]
                         {submit_jobs,run,check}
```


