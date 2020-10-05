import torch
from torchvision import datasets, transforms

class Mnist:
    def __init__(self, batch_size, norm_dgt=9, nu=0.0):
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_mnist = datasets.MNIST("../data", train=True, download=True, transform=dataset_transform)
        test_mnist = datasets.MNIST("../data", train=False, download=True, transform=dataset_transform)
        test_mnist_anom = datasets.MNIST("../data", train=False, download=True, transform=dataset_transform)

        train_image, train_label = train_mnist.train_data, train_mnist.train_labels
        test_image, test_label = test_mnist.test_data, test_mnist.test_labels

        train_normal = torch.stack(
            [train_image[key] for (key, label) in enumerate(train_label) if label.int().item() == norm_dgt])
        train_anom = torch.stack(
            [train_image[key] for (key, label) in enumerate(train_label) if label.int().item() != norm_dgt])
        test_normal = torch.stack(
            [test_image[key] for (key, label) in enumerate(test_label) if label.int().item() == norm_dgt])
        test_anom = torch.stack(
            [test_image[key] for (key, label) in enumerate(test_label) if label.int().item() != norm_dgt])
        train_label_normal = torch.stack(
            [train_label[key] for (key, label) in enumerate(train_label) if label.int().item() == norm_dgt])
        train_label_anom = torch.stack(
            [train_label[key] for (key, label) in enumerate(train_label) if label.int().item() != norm_dgt])
        test_label_normal = torch.stack(
            [test_label[key] for (key, label) in enumerate(test_label) if label.int().item() == norm_dgt])
        test_label_anom = torch.stack(
            [test_label[key] for (key, label) in enumerate(test_label) if label.int().item() != norm_dgt])

        # build training set with a mix of normal and anom data, proportion is defined by 'nu'
        perm = torch.randperm(train_anom.size(0))
        samples = perm[:int(nu * train_normal.size(0))]
        train_anom = train_anom[samples]
        train_label_anom = train_label_anom[samples]
        
        train_mnist.train_data = torch.cat((train_normal, train_anom), dim=0)
        train_mnist.train_labels = torch.cat((train_label_normal, train_label_anom), dim=0)

        # test set is easier, we keep it split
        test_mnist.test_data = test_normal
        test_mnist_anom.test_data = test_anom
        test_mnist.test_labels = test_label_normal
        test_mnist_anom.test_labels = test_label_anom

        self.train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batch_size, shuffle=False)
        self.test_anom_loader = torch.utils.data.DataLoader(test_mnist_anom, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    mnist_obj = Mnist(batch_size=32, norm_dgt=9, nu=0.0)
    print(mnist_obj)