import torch
import torch.nn.functional as F
from torch import device
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Food101
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Her process için GPU ayarla


class Trainer:
    def __init__(self, gpu_id: int,
                 dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_every: int  # bool yerine int olmalı
                 ):
        self.dataloader = dataloader
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])  # Modeli GPU'ya taşı
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu_id = gpu_id
        self.optimizer = optimizer
        self.save_every = save_every

    def train(self, max_epoch):
        self.model.train()
        for epoch in range(max_epoch):
            self.run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"checkpoint_epoch_{epoch}.pt"  # Daha açıklayıcı dosya adı
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def run_epoch(self, epoch):
        b_sz = len(next(iter(self.dataloader))[0])
        self.dataloader.sampler.set_epoch(epoch)
        print(f'[GPU {self.gpu_id}] Epoch {epoch} started')

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.gpu_id), target.to(self.gpu_id)
            loss = self.run_batch(data, target)
            total_loss += loss
            num_batches += 1

            if batch_idx % 100 == 0:  # Her 100 batch'te bir log
                print(f'[GPU {self.gpu_id}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')

        avg_loss = total_loss / num_batches
        print(f'[GPU {self.gpu_id}] Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

    def run_batch(self, data, target):
        logits = self.model(data)
        loss = self.criterion(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()  # Loss değerini return et


def get_model_optimizer():
    # Food101 için uygun bir model oluştur
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 512),  # Food101 için 224x224 RGB görüntüler
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 101)  # Food101'de 101 sınıf var
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer daha iyi
    return model, optimizer


def main(rank, world_size, save_every):
    ddp_setup(rank, world_size)

    # Transform pipeline'ını düzelt
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Görüntüleri yeniden boyutlandır
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
    ])

    try:
        dataset_train = Food101(root='./data', split='train', transform=transform, download=True)
        dataloader = DataLoader(
            dataset_train,
            batch_size=16,  # Daha küçük batch size bellek için
            shuffle=False,
            sampler=DistributedSampler(dataset_train),
            num_workers=2,  # Daha hızlı veri yükleme
            pin_memory=True
        )

        model, optimizer = get_model_optimizer()
        trainer = Trainer(
            gpu_id=rank,
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            save_every=save_every
        )
        trainer.train(max_epoch=10)  # Test için daha az epoch

    except Exception as e:
        print(f"GPU {rank} encountered error: {e}")
    finally:
        destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("CUDA is not available!")
        exit(1)

    save_every = 2  # Her 2 epoch'ta bir kaydet
    print(f"Starting training with {world_size} GPUs")
    mp.spawn(main, args=(world_size, save_every), nprocs=world_size)