import torchvision.transforms as T

def make_train_transformations():
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([224, 224]),
        T.Normalize()
    ])