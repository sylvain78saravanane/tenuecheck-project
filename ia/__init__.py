def __init__(self):
    self.use_custom_model = False
    self.dresscode_model = None
    self.person_model = None

    # Chemin absolu vers dresscode_yolo.pt (dans ia/)
    custom_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dresscode_yolo.pt"
    )