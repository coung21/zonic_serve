import importlib

def import_class(class_path: str):
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to import class '{class_path}': {e}")



if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    model_class = import_class("src.model.dummy.DummyModel")
    print(model_class)