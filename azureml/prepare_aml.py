from azureml.core import Workspace, Environment

if __name__ == '__main__':
    ws = Workspace.from_config('.config/aml_workspace.json')

    # Create AML Environment
    env = Environment("minerl-th")
    env.docker.base_image = None
    env.docker.base_dockerfile = "Dockerfile"
    env.python.user_managed_dependencies = True
    env.register(workspace=ws)

    # Create Compute Cluster
