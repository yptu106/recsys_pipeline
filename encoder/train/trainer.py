import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Union

from encoder.data.dataset import TripletDatasetWithUserEmbeddings
from encoder.utils.triplet_sampling import sample_triplets, build_triplets_from_val_df
from encoder.utils.embedding_ops import build_user_embedding_cache, encode_texts


class EncoderTrainer:
    def __init__(
        self, encoder, device, 
        optimizer, loss_fn, 
        save_path, patience,
        batch_size=32,
        log_path=None,
        config: dict = None,
        config_path=None,  # expect to end with .yaml or .yml
        val_freq: int = 1
    ):
        self.encoder = encoder
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.save_path = save_path
        self.patience = patience
        self.log_path = log_path
        self.config_path = config_path
        self.val_freq = val_freq

        if log_path:
            with open(log_path, 'w') as f:
                f.write("epoch,avg_loss,val_loss,best_so_far\n")
        
        if config_path and config:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)

    def validate(
        self,
        user_to_items: dict,
        item_id_to_text: dict,
        val_triplets: list[tuple[str, str, str]],
        batch_size: int = 32
    ) -> float:
        self.encoder.eval()

        user_embeddings = build_user_embedding_cache(
            self.encoder, user_to_items, item_id_to_text
        )
        dataset = TripletDatasetWithUserEmbeddings(
            user_embeddings, item_id_to_text, val_triplets
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_val_loss = 0.0
        with torch.no_grad():
            for user_emb_batch, pos_text_batch, neg_text_batch in tqdm(dataloader, desc="Validating"):
                user_emb_batch = user_emb_batch.to(self.device)

                pos_emb_batch = encode_texts(self.encoder, list(pos_text_batch), self.device)
                neg_emb_batch = encode_texts(self.encoder, list(neg_text_batch), self.device)

                loss = self.loss_fn(user_emb_batch, pos_emb_batch, neg_emb_batch)
                total_val_loss += loss.item()

        return total_val_loss / len(dataloader)


    def train(
        self, 
        user_to_items: dict,
        item_id_to_text: dict,
        train_triplets: list[tuple[str, str, str]],
        num_epochs: int = 10,
        val_triplets: Optional[list[tuple[str, str, str]]] = None,
    ):
        best_loss = float('inf') # track best validation loss or training loss
        best_epoch = 0
        patience_counter = 0

        print("=== Training Started ===")
        print(f"Model: {self.encoder.__class__.__name__}")
        print(f"Save path: {self.save_path}")
        print(f"Num epochs: {num_epochs}")
        print(f"Patience: {self.patience}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n› Epoch {epoch}/{num_epochs}")

            # rebuild user embeddings and dataloader
            user_embeddings = build_user_embedding_cache(self.encoder, user_to_items, item_id_to_text)
            dataset = TripletDatasetWithUserEmbeddings(
                user_embeddings=user_embeddings,
                item_id_to_text=item_id_to_text,
                triplets=train_triplets
            )
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            print(f"Training on {len(dataloader)} batches with batch size {self.batch_size}")

            total_loss = 0.0
            self.encoder.train()

            for user_emb_batch, pos_text_batch, neg_text_batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                # move user embeddings to device 
                user_emb_batch = user_emb_batch.to(self.device)

                # encode positive and negative item texts
                pos_emb_batch = encode_texts(self.encoder, list(pos_text_batch), self.device)
                neg_emb_batch = encode_texts(self.encoder, list(neg_text_batch), self.device)

                loss = self.loss_fn(user_emb_batch, pos_emb_batch, neg_emb_batch)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                del user_emb_batch, pos_text_batch, neg_text_batch, pos_emb_batch, neg_emb_batch, loss
                torch.cuda.empty_cache()  # Optional, reduces fragmentation risk

            avg_loss = total_loss / len(dataloader)
            val_loss = None

            loss_to_track = None
            is_tracking_epoch = False

            if val_triplets: # use validation loss if available
                if epoch % self.val_freq == 0:
                    val_loss = self.validate(
                        user_to_items=user_to_items,
                        item_id_to_text=item_id_to_text,
                        val_triplets=val_triplets,
                        batch_size=self.batch_size
                    )
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
                # torch.save(self.encoder.state_dict(), self.save_path)
                self.encoder.save(self.save_path)  # assuming the encoder has a save method

                print(f"   New best model saved at epoch {epoch} with loss {best_loss:.4f}")
            elif is_tracking_epoch: # only increment patience if this epoch was tracked
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"    Early stopping triggered. No improvement for {patience_counter} epochs.")
                    break
        
        print(f"\n=== Training Finished ===")
        print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
        print(f"Model saved to: {self.save_path}")