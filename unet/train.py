import torch


def validate_model(model, val_loader, loss_fn, device):
    """
    在验证集上评估模型性能
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # 关闭梯度计算
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model.train()  # 重新设置为训练模式
    return avg_loss


def train(
    model, train_dataloader, device, val_dataloader, criterion, optimizer, num_epochs
):
    model.to(device)
    # 更改学习率调度器为更敏感的参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5, min_lr=1e-7
    )

    # 添加早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # 计算损失
            loss = criterion(output, target)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Training Loss: {loss.item():.4f}"
                )

        val_loss = validate_model(model, val_dataloader, criterion, device)

        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break
