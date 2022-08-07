from azureml.core import Workspace, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Root directory path for config and keys')
    args = parser.parse_args()

    ws = Workspace.from_config(f'{args.config_path}/aml_workspace.json')

    # Create AML Environment
    env = Environment.from_dockerfile('minerl', '../Dockerfile')
    env.register(workspace=ws)

    # Create Compute Cluster
    cluster_name = 'BasicK80'
    try:
        cluster = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        # To use a different region for the compute, add a location='<region>' parameter
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                               max_nodes=1)
        cluster = ComputeTarget.create(ws, cluster_name, compute_config)

    cluster.wait_for_completion(show_output=True)
