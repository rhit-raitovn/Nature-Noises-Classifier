import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import random
from collections import Counter

class AudioSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, specs, labels, augment=False):
        self.specs = specs
        self.labels = labels
        self.augment = augment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=10)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        spec = self.specs[idx]
        if self.augment:
            spec = spec.clone()
            if random.random() < 0.5:
                spec = self.freq_mask(spec)
            if random.random() < 0.5:
                spec = self.time_mask(spec)
        return spec, label


def collate_fn_test(batch):
    return batch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, output_size, input_shape=(1, 64, 130)):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 256)

        self.dropout = nn.Dropout(0.3)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.get_flattened_size(dummy_input)
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def get_flattened_size(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.pool(x)
        x = self.res_block2(x)
        x = self.pool(x)
        x = self.res_block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_model(output_size, learning_rate=0.0005):
    model = CNNModel(output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return model, criterion, optimizer, scheduler


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            parts_list, label = batch[0]
            parts_batch = torch.stack(parts_list).to(device)
            outputs = model(parts_batch)
            _, predicted = torch.max(outputs.data, 1)
            mode_prediction = Counter(predicted.cpu().tolist()).most_common(1)[0][0]
            label_value = label.item() if torch.is_tensor(label) else label
            correct += (mode_prediction == label_value)
            total += 1

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    model.train()


def train_model(model, device, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model(model, test_loader, device)

    print("Training complete.")

if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_spectrograms_train = torch.load("preprocessed_spectrograms_train_10000.pt", weights_only=False)
    preprocessed_labels_train = torch.load("preprocessed_labels_train_10000.pt", weights_only=False)
    preprocessed_spectrograms_test = torch.load("preprocessed_spectrograms_test_1000.pt", weights_only=False)
    preprocessed_labels_test = torch.load("preprocessed_labels_test_1000.pt", weights_only=False)

    dataset_train = AudioSpectrogramDataset(preprocessed_spectrograms_train, preprocessed_labels_train, augment=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)

    dataset_test = AudioSpectrogramDataset(preprocessed_spectrograms_test, preprocessed_labels_test, augment=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn_test)

    num_classes = len(set(preprocessed_labels_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion, optimizer, scheduler = create_model(num_classes)
    model.to(device)

    train_model(model, device, criterion, optimizer, scheduler, dataloader_train, dataloader_test, num_epochs=20)
    evaluate_model(model, dataloader_test, device)
