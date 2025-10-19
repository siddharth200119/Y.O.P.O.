from yopo import YOPO

if __name__ == '__main__':
    print("Testing YOPO class...")
    # Instantiate YOPO with a model size to trigger the download logic
    yopo_instance = YOPO(modal_size='n')
    print("YOPO instance created successfully.")
    # The model is loaded in the constructor. We can check the model object.
    if yopo_instance.modal:
        print("Model loaded successfully.")
        # The ultralytics YOLO object doesn't have a direct `path` attribute,
        # but ckpt_path is a good indicator.
        print(f"Model path: {yopo_instance.modal.ckpt_path}")
    else:
        print("Model loading failed.")