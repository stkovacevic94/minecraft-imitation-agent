import azureml.core as aml
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Root directory path for config and keys')
    args = parser.parse_args()

    # Load AML Workspace
    ws = aml.Workspace.from_config(f'{args.config_path}/aml_workspace.json')
    with open(f"{args.config_path}/auth_keys.json") as auth_keys_file:
        auth_keys = json.load(auth_keys_file)

    experiment_name = "master-thesis"

    # Load Env and setup Env variables
    env = aml.Environment.get(workspace=ws, name='minerl-th')
    env.environment_variables['WANDB_API_KEY'] = auth_keys['wandb']
    env.environment_variables['WANDB_ENTITY'] = 'stkovacevic94'
    env.environment_variables['WANDB_PROJECT'] = experiment_name

    experiment = aml.Experiment(workspace=ws, name=experiment_name)

    config = aml.ScriptRunConfig(
        source_directory='../',
        command=[
             'xvfb-run -a python',
             'run_bco.py',
             '--data_path', "./dataset",
             '--batch_size', '64',
             '--max_episodes', '500'
        ],
        docker_runtime_config=aml.runconfig.DockerConfiguration(use_docker=True),
        environment=env,
        compute_target=aml.ComputeTarget(workspace=ws, name='BasicK80'))

    train_run = experiment.submit(config)
