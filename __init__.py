import glob
import importlib.util
import os

extension_folder = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

pyPath = os.path.join(extension_folder, 'nodes')

def loadCustomNodes():
    files = glob.glob(os.path.join(pyPath, "*Node.py"), recursive=True)
    for file in files:
        file_relative_path = file[len(extension_folder):]
        model_name = file_relative_path.replace(os.sep, '.')
        model_name = os.path.splitext(model_name)[0]
        module = importlib.import_module(model_name, __name__)
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

loadCustomNodes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
