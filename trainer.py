import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

# --- Mock Components for Standalone Execution ---
# In a real project, these would be in separate files.
class MockEncoder(nn.Module):
    """A mock encoder model for demonstration."""
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Core Trainer Class ---
class DualEncoderTrainer:
    """
    Implements the training logic for a dual-encoder model using momentum contrast (MoCo),
    advanced loss functions, and mixed-precision training.
    """
    def __init__(
        self,
        model_q: nn.Module,
        queue_size: int = 65536,
        embedding_dim: int = 128,
        tau: float = 0.07,
        m: float = 0.999,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.q = model_q.to(self.device)
        self.k = copy.deepcopy(model_q).to(self.device)
        self.tau = tau
        self.m = m
        self.embedding_dim = embedding_dim

        # Deactivate gradients for the key encoder
        for param in self.k.parameters():
            param.requires_grad = False

        self.register_queue(queue_size, embedding_dim)

        self.scaler = GradScaler()
        self.opt = torch.optim.AdamW(self.q.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # A scheduler is crucial for stable training
        self.scheduler = CosineAnnealingLR(self.opt, T_max=10000) # T_max should be total steps

        print("DualEncoderTrainer initialized.")
        print(f" - Device: {self.device}")
        print(f" - Queue Size: {queue_size}")
        print(f" - Embedding Dim: {self.embedding_dim}")

    def register_queue(self, size, dim):
        """Initializes the negative sample queue."""
        self.register_buffer('queue', torch.randn(size, dim, device=self.device))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def register_buffer(self, name, tensor):
        """Helper to register a buffer to the class."""
        setattr(self, name, tensor)

    @torch.no_grad()
    def _momentum_update(self):
        """Applies the momentum update to the key encoder."""
        for param_q, param_k in zip(self.q.parameters(), self.k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Updates the queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Ensure the batch fits
        assert self.queue.shape[0] % batch_size == 0

        # Replace the oldest keys in the queue
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.queue.shape[0]


    def info_nce_loss(self, q_embeds, k_pos_embeds):
        """Calculates the core InfoNCE loss."""
        q = F.normalize(q_embeds, dim=1)
        k_pos = F.normalize(k_pos_embeds, dim=1)

        # Positive similarity
        l_pos = (q * k_pos).sum(dim=1, keepdim=True)

        # Negative similarity (with the queue)
        l_neg = torch.matmul(q, self.queue.clone().detach().t())

        # Combine and apply temperature scaling
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau

        # Labels: the first logit (positive) is the correct one
        labels = torch.zeros(q.size(0), dtype=torch.long, device=self.device)

        return F.cross_entropy(logits, labels)

    def train_step(self, batch):
        """Performs a single training step."""
        # Assumes batch is a dict with 'q' and 'k' keys for the two views of the data
        x_q, x_k = batch['q'].to(self.device), batch['k'].to(self.device)

        self.opt.zero_grad()

        with autocast():
            # 1. Compute query embeddings
            q_emb = self.q(x_q)

            # 2. Compute key embeddings with the key encoder
            with torch.no_grad():
                k_emb = self.k(x_k)

            # 3. Calculate core InfoNCE loss
            loss_info = self.info_nce_loss(q_emb, k_emb)

            # --- Placeholder for Advanced Losses ---
            # loss_arcface = self.arcface_loss(q_emb, labels) # Requires labels
            # loss_triplet = self.triplet_loss(q_emb, k_emb, negatives) # Requires hard negative mining
            # loss_center = self.center_loss(q_emb, labels) # Requires labels and center tracking
            # loss_ortho = self.orthogonality_regularizer(self.q.parameters())

            # Combine losses with weighting factors (lambda)
            # loss = loss_info + lambda1 * loss_arcface + lambda2 * loss_triplet ...
            loss = loss_info

        # 4. Backpropagation with GradScaler
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.scheduler.step()

        # 5. Update key encoder and queue
        self._momentum_update()
        self._dequeue_and_enqueue(k_emb)

        return loss.item()

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 64
    IN_DIM = 256
    EMBEDDING_DIM = 128
    QUEUE_SIZE = 4096

    # 1. Create model and trainer
    encoder = MockEncoder(in_dim=IN_DIM, out_dim=EMBEDDING_DIM)
    trainer = DualEncoderTrainer(
        model_q=encoder,
        queue_size=QUEUE_SIZE,
        embedding_dim=EMBEDDING_DIM
    )

    # 2. Create a dummy dataloader
    # In a real scenario, this would load augmented pairs (e.g., two different crops of an image)
    dummy_dataset = [{'q': torch.randn(IN_DIM), 'k': torch.randn(IN_DIM)} for _ in range(BATCH_SIZE * 10)]
    dataloader = torch.utils.data.DataLoader(
        dummy_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: {'q': torch.stack([i['q'] for i in x]), 'k': torch.stack([i['k'] for i in x])}
    )

    # 3. Training loop
    print("\nStarting mock training loop...")
    for epoch in range(3):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            loss = trainer.train_step(batch)
            total_loss += loss
            if i % 5 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss:.4f}, LR: {trainer.scheduler.get_last_lr()[0]:.6f}")
        print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f} ---\n")

    print("Mock training complete.")
    # You can now save the query encoder:
    # torch.save(trainer.q.state_dict(), 'final_encoder.pth')