# project-init
A code to init the project and experiment for easy-to-use 

## Set variables

We can set 5 variables in the command line. 

- `c` (mandatory): the path of the configuration file to be used.
- `b` (mandatory): the path of the base directory to be used for the whole project running, set `base_path`.
- `p` (mandatory): the base name of the project to be running, set `base_project_name`.
- `u` (mandatory): the keys of the config that are required by the user to be used as the user project name, set `user_project_name`.
    - Here is an example of setting `-u`: `data|data_name;model|model_name;train|epoch;` where `;` is to separate different blocks of the configuration file and `|` is to present each layer of setting under the block. `data|data_name` means using the `data_name` under the `data` block. 
- `r` (optional): the name of the csv file to save running information.
- `l` (optional): the logging level.

Thus, we can get the `base_name` that is the base folder name of `base_path`. The `fix_project_name` is `{base_project_name}--{data_name}--{model_name}`. The project name is `{fix_project_name}--{user_project_name}`. The `project_path` is `{base_path}/{project_name}`. The `log_id` is the unique id for each running.



## Structure
    .
    ├── base_path                                       # Config.args.b
    └──── data (data_path)                              # data path
    └──── {fix_project_name}--{user_project_name}     # Config.args.p: project_name
    └───────── {log_id}                                 # Unique log id
    └───────────── checkpoints                          # checkpoints
    └───────────── results                              # results
    └───────────── loggings                             # loggings
    └───────────── visualizations                       # visualizations
    ├── experiments.csv                                 # Config.args.r     

where `project_name` is `{fix_project_name}--{user_project_name}`

See `set_records` function of the `Config()` to check the content of the `experiments.csv`. 


## Wandb
We will also create a wandb project with the structure close to above structure. The entity will be set by the user in the function call. When the entity is different from the `base_name`, the wandb project name will be `{base_name}---{project_name}`. Otherwise, we will use the `base_name` as the entity.

## Example

- With user's desired keys
```bash
$ python examples/test.py -c configs/test.yml -b InitTest -p SWork -u "environment|dotenv_path;train|epoch" -r experiments.csv
```

This command will create a folder named `InitTest` under the root path. Under this folder, we will have a folder named `SWork--MATH--TestModel--.env--10` where `.env` and `10` are the values defined by the keys of `-u`. Under this project folder, we will have a folder with the unique id to store all outputs of the running. The running information will be saved in the `experiments.csv` file.

- Without user's desired keys
```bash
$ python examples/test.py -c configs/test.yml -b InitTest -p SWork -r experiments.csv
```



