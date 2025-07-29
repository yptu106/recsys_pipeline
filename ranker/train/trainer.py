import yaml
import torch
from tqdm import tqdm
from contextlib import redirect_stdout, nullcontext

class RankerTrainer:
    def __init__(
        self, model, device, 
        optimizer, 
        save_path, patience, 
        loss_fn=None,
        log_path=None, 
        config: dict = None,
        config_path=None, # expect to end with .yaml or .yml
        val_freq: int = 1
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.save_path = save_path
        self.patience = patience
        self.log_path = log_path
        self.config_path = config_path
        self.val_freq = val_freq
        self.loss_fn = loss_fn

        if log_path:
            with open(log_path, 'w') as f:
                f.write("epoch,avg_loss,val_loss,best_so_far\n")

        if config_path and config:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)

    def _compute_loss(self, batch, mode):
        if mode == "pairwise":
            u, pos_i, neg_i = batch
            u, pos_i, neg_i = u.to(self.device), pos_i.to(self.device), neg_i.to(self.device)
            pos_score = self.model(u, pos_i)
            neg_score = self.model(u, neg_i)
            return self.loss_fn(pos_score, neg_score)
        
        elif mode == "pointwise":
            u, i, label = batch
            u, i, label = u.to(self.device), i.to(self.device), label.to(self.device)
            score = self.model(u, i)
            return self.loss_fn(score, label)
        
        elif mode == "contextual_pairwise":
            hist, pos_i, neg_i, _history_lens = batch
            hist, pos_i, neg_i = hist.to(self.device), pos_i.to(self.device), neg_i.to(self.device)
            pos_score = self.model(hist, pos_i)
            neg_score = self.model(hist, neg_i)
            return self.loss_fn(pos_score, neg_score)

        else:
            raise ValueError(f"Unknown mode: {mode}")


    def validate(self, dataloader, loss_fn=None, mode="pairwise"):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                loss = self._compute_loss(batch, mode)
                total_loss += loss.item()

                # Explicitly free tensors
                del loss
                torch.cuda.empty_cache()  # Optional, reduces fragmentation risk

        return total_loss / len(dataloader)

    def train(self, dataloader, num_epochs=10, val_dataloader=None, loss_fn=None, mode="pairwise"):
        best_loss = float('inf') # track best validation loss or training loss
        best_epoch = 0
        patience_counter = 0
        self.loss_fn = loss_fn or self.loss_fn

        print("=== Training Started ===")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Save path: {self.save_path}")
        print(f"Num epochs: {num_epochs}")
        print(f"Patience: {self.patience}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n› Epoch {epoch}/{num_epochs}")

            self.model.train()
            total_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                loss = self._compute_loss(batch, mode)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            val_loss = None

            loss_to_track = None
            is_tracking_epoch = False

            if val_dataloader: # use validation loss if available
                if epoch % self.val_freq == 0:
                    val_loss = self.validate(val_dataloader, self.loss_fn, mode)
                    loss_to_track = val_loss
                    is_tracking_epoch = True
            else: # fallback to training loss
                loss_to_track = avg_loss
                is_tracking_epoch = True

            print(f"[Epoch {epoch}] Train Loss = {avg_loss:.4f}" + (f", Val Loss = {val_loss:.4f}" if val_loss is not None else ""))

            is_best = False
            if is_tracking_epoch:
                is_best = loss_to_track < best_loss

            if self.log_path:
                with open(self.log_path, 'a') as f:
                    if val_loss is not None:
                        f.write(f"{epoch},{avg_loss:.6f},{val_loss:.6f},{'yes' if is_best else 'no'}\n")
                    else:
                        f.write(f"{epoch},{avg_loss:.6f},,{ 'yes' if is_best else 'no'}\n")

            if is_best:
                best_loss = loss_to_track
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"   New best model saved at epoch {epoch} with loss {best_loss:.4f}")
            elif is_tracking_epoch: # only increment patience if this epoch was tracked
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"    Early stopping triggered. No improvement for {patience_counter} epochs.")
                    break
        
        print(f"\n=== Training Finished ===")
        print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
        print(f"Model saved to: {self.save_path}")