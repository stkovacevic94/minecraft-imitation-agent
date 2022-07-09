import azureml.core as aml
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Root directory path for config and keys')
    args = parser.parse_args()

    ws = aml.Workspace.from_config(f'{args.config_path}/aml_workspace.json')
    with open(f"{args.config_path}/auth_keys.json") as auth_keys_file:
        auth_keys = json.load(auth_keys_file)

    experiment_name = "master-thesis"

    env = aml.Environment.get(workspace=ws, name='minerl-th')
    env.environment_variables['WANDB_API_KEY'] = auth_keys['wandb']
    env.environment_variables['WANDB_ENTITY'] = 'skovacevic94'
    env.environment_variables['WANDB_PROJECT'] = experiment_name
    env.environment_variables['CUDA_LAUNCH_BLOCKING'] = '1'

    script_config = aml.ScriptRunConfig(
        source_directory="./src",
        script='main.py',
        arguments=[
                '--data_path', "./dataset",
                '--lr', 0.0001,
                '--batch_size', 128,
                '--max_epoch', 100,
                '--seed', 42],
        environment=env,
        compute_target=aml.ComputeTarget(workspace=ws, name='BasicK80')
    )

    run = aml.Experiment(workspace=ws, name=experiment_name).submit(config=script_config)
