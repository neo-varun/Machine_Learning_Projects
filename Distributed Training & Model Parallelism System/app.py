import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class LargeDataset(Dataset):
    def __init__(self, size=1_000_000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(100)
        y = torch.sum(x)
        return x, y


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, epochs=15, batch_size=128):

    print(f"Running on rank {rank}")

    setup(rank, world_size)

    pin_memory = torch.cuda.is_available()

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    dataset = LargeDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=pin_memory,
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        start_time = time.time()
        epoch_loss = 0.0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        end_time = time.time()

        if rank == 0:
            print(
                f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Time: {end_time - start_time:.2f}s"
            )

    cleanup()


def main():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2

    print(f"Using {world_size} processes")

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
