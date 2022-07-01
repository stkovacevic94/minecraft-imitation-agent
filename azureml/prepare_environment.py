from azureml.core import Workspace, Environment

if __name__ == '__main__':
    ws = Workspace.from_config('.config/aml_workspace.json')

    env = Environment.from_conda_specification('minerl-th', 'environment.yml')
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04'
    env.register(ws)
