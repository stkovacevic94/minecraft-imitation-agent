from azureml.core import Workspace, Dataset

ws = Workspace.from_config('./.config/azureml_workspace.json')

datasets.CIFAR10(root='./data', train=True, download=True)
datasets.CIFAR10(root='./data', train=False, download=True)

datastore = ws.get_default_datastore()
datastore.upload(src_dir='./data', target_path='datasets/cifar10', overwrite=True)
Dataset.File.from_files(path=(datastore, 'datasets/cifar10')).register(workspace=ws, name='CIFAR10')