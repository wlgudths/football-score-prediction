import os
import joblib
from tqdm import tqdm
import torch
from utils.logger import logger

def dl_train(model,
             train_loader,
             val_loader,
             criterion,
             optimizer,
             scheduler,
             device,
             config):
    '''
    '''
    logger.info('[Start training] 훈련 시작')

    train_config = config['train']
    epochs = train_config['max_epochs']
    valid_interval = train_config['valid_interval']
    delta = train_config['early_stop']['delta']
    patience = train_config['early_stop']['patience']
    
    save_dir = config['save']['output_path']
    save_path = os.path.join(save_dir, 'best_model.pth')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f'[Epoch {epoch+1}/{epochs}] Training', leave=False)

        for batch_x, batch_y in train_loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        if (epoch + 1) % valid_interval == 0:
            model.eval()
            val_loss = 0.0
            val_loop = tqdm(val_loader, desc=f'[Epoch {epoch+1}/{epochs}] Validation', leave=False)

            with torch.no_grad():
                for batch_x, batch_y in val_loop:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())

                avg_val_loss = val_loss / len(val_loader)
                logger.info(f'Epoch [{epoch+1}/{epochs}] | Train_loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss + delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
                
                logger.info(f'New best Model saved (Val Loss: {best_val_loss:.4f})')
                torch.save(best_model_state, save_path)

            else:
                epochs_no_improve += 1
                logger.info(f'No improvement for {epochs_no_improve} epochs')

                if epochs_no_improve >= patience:
                    logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                    break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)    
    
    return model


def ml_train(X_train,
             y_train,
             X_val,
             y_val,
             model,
             criterion,
             config):
    
    logger.info('[Start training] 훈련 시작')

    model.fit(X_train, y_train)

    y_preds = model.predict(X_val)
    val_loss = criterion(y_val, y_preds)

    logger.info(f'[ML Training Completed] Validation MSE: {val_loss:.4f}')

    save_dir = config['save']['output_path']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pkl')
    joblib.dump(model, save_path)
    
    logger.info(f'[Model Saved] 경로: {save_path}')

    return model
