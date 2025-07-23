import yaml
import torch
from tqdm import tqdm
from contextlib import redirect_stdout, nullcontext

class RankerTrainer:
    def __init__(
        self, model, device, 
        optimizer, loss_fn, 
        save_path, patience, 
        log_path=None, 
        config: dict = None,
        config_path=None # expect to end with .yaml or .yml
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.patience = patience
        self.log_path = log_path
        self.config_path = config_path

        if log_path:
            with open(log_path, 'w') as f:
                f.write("epoch,avg_loss,best_so_far\n")

        if config_path and config:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)

    def train(self, dataloader, num_epochs):
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        print("=== Training Started ===")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Save path: {self.save_path}")
        print(f"Num epochs: {num_epochs}")
        print(f"Patience: {self.patience}\n")

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0

            for u, pos_i, neg_i in tqdm(dataloader, desc=f"Epoch {epoch}"):
                u, pos_i, neg_i = u.to(self.device), pos_i.to(self.device), neg_i.to(self.device)

                pos_score = self.model(u, pos_i)
                neg_score = self.model(u, neg_i)

                loss = self.loss_fn(pos_score, neg_score)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            is_best = avg_loss < best_loss

            print(f"[Epoch {epoch}] Loss = {avg_loss:.4f}")

            if self.log_path:
                with open(self.log_path, 'a') as f:
                    f.write(f"{epoch},{avg_loss:.6f},{'yes' if is_best else 'no'}\n")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"   New best model saved at epoch {epoch} with loss {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"    Early stopping triggered. No improvement for {patience_counter} epochs.")
                    break
        
            print(f"\n=== Training Finished ===")
            print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
            print(f"Model saved to: {self.save_path}")