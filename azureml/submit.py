import azureml.core as aml
import json

ws = aml.Workspace.from_config('./.config/azureml_workspace.json')
with open("./.config/auth_keys.json") as auth_keys_file:
    auth_keys = json.load(auth_keys_file)

experiment_name = "CIFAR10"

env = aml.Environment.get(workspace=ws, name='PyTorch-1.9.0')
env.environment_variables['WANDB_API_KEY'] = auth_keys['wandb']
env.environment_variables['WANDB_ENTITY'] = 'skovacevic94'
env.environment_variables['WANDB_PROJECT'] = experiment_name

script_config = aml.ScriptRunConfig(
    source_directory="./src",
    script='main.py',
    arguments=[
            '--data_path', aml.Dataset.get_by_name(workspace=ws, name='CIFAR10').as_mount(),
            '--lr', 0.001,
            '--batch_size', 64,
            '--max_epoch', 100,
            '--seed', 42],
    environment=env,
    compute_target=aml.ComputeTarget(workspace=ws, name='BasicK80')
)

run = aml.Experiment(workspace=ws, name=experiment_name).submit(config=script_config)
