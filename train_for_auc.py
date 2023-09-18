import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import Response_WSI_Gene_Dataset
from model_finger import Porpoise_MMF
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelBinarizer

# 定义超参数
learning_rate = 0.001
num_epochs = 10
batch_size = 1
early_stopping_patience = 3  # 设置提前停止的耐心
best_auc = 0.0 
val_auc = 0.0
# 创建数据集和数据加载器
dataset = Response_WSI_Gene_Dataset(csv_path='../PORPOISE-master/datasets_csv/tcga_gbmlgg_trian_new_finger_clean.csv.zip')
dataset_size = len(dataset)
validation_split = 0.2
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

# 随机打乱数据集
np.random.shuffle(indices)

# 分割数据集
train_indices, val_indices = indices[split:], indices[:split]

# 创建训练集和验证集的数据加载器
batch_size = 1
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# 使用tqdm显示数据加载进度
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, shuffle=False)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, shuffle=False)

# 创建神经网络模型
model_dict = {
    'omic_input_dim': 18294,
    'path_input_dim': 1024,
    'finger_input_dim': 1024,
    'dropout': 0.25,
    'n_classes': 4,
    'scale_dim1': 8,
    'scale_dim2': 8,
    'gate_path': 1,
    'gate_omic': 1,
    'skip': True,
    'dropinput': 0.10,
    'size_arg': 'small'
}
model = Porpoise_MMF(**model_dict)

# 使用DataParallel包装模型以进行多GPU训练
model = nn.DataParallel(model)

# 将模型移动到GPU
model.to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建学习率调度器
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # 每3个epoch将学习率缩小为原来的0.1

# 初始化提前停止相关变量
best_auc = 0.0
no_improvement_count = 0

# 创建一个空的DataFrame来保存结果
results_df = pd.DataFrame(columns=['Case_ID', 'Label', 'Predicted'])

# 创建TensorBoard写入器
writer = SummaryWriter()

# 初始化LabelBinarizer
label_binarizer = LabelBinarizer()
label_binarizer.fit(dataset.label_col.unique())

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    scores = []  # 用于保存每个验证样本的预测分数
    labels = []  # 用于保存每个验证样本的真实标签
    
    # 使用tqdm包装训练数据加载器
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        slide_id, label, gene, slide_feature, finger = batch
        # 将数据移动到GPU
        gene = gene.to('cuda').squeeze(dim=0).to(torch.float32)
        slide_feature = slide_feature.to('cuda').squeeze(dim=0)
        finger = finger.to('cuda').to(torch.float32)
        label = label.to('cuda')
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(x_path=slide_feature, x_omic=gene, x_finger=finger)
        
        # 计算损失
        loss = criterion(outputs, label)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 计算softmax得分
        softmax_scores = torch.softmax(outputs, dim=1)
        scores.extend(softmax_scores.tolist())
        labels.extend(label.tolist())
        
        total_loss += loss.item()
    
    # 计算平均损失
    average_loss = total_loss / len(train_loader)
    
    # 提前停止检查
    if val_auc > best_auc:
        best_auc = val_auc
        no_improvement_count = 0
        # 保存训练好的模型
        torch.save(model.state_dict(), f'your_auc_model_weights_epoch_{epoch}.pth')
    else:
        no_improvement_count += 1
    
    # 执行验证步骤
    model.eval()  # 切换模型到评估模式
    val_loss = 0.0
    val_scores = []
    val_labels = []
    slide_ids = []
    
    with torch.no_grad():
        # 使用tqdm包装验证数据加载器
        for val_batch in tqdm(val_loader, desc=f'Validation - Epoch {epoch+1}/{num_epochs}'):
            slide_id, label, gene, slide_feature, finger = val_batch
            gene = gene.to('cuda').squeeze(dim=0).to(torch.float32)
            slide_feature = slide_feature.to('cuda').squeeze(dim=0)
            finger = finger.to('cuda').to(torch.float32)
            label = label.to('cuda')
            
            outputs = model(x_path=slide_feature, x_omic=gene, x_finger=finger)
            val_loss += criterion(outputs, label).item()
            
            # 计算softmax得分
            softmax_scores = torch.softmax(outputs, dim=1)
            val_scores.extend(softmax_scores.tolist())
            val_labels.extend(label.tolist())
            slide_ids.extend(slide_id)
    
    # 计算验证集上的平均损失
    val_average_loss = val_loss / len(val_loader)
    
    # 计算AUC
    val_labels = label_binarizer.transform(val_labels)  # 将验证集标签进行独热编码
    val_auc = roc_auc_score(val_labels, val_scores, average='weighted')
    
    # 在TensorBoard中记录验证损失和AUC
    writer.add_scalar('Validation Loss', val_average_loss, epoch)
    writer.add_scalar('Validation AUC', val_auc, epoch)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_average_loss:.4f}, Validation AUC: {val_auc:.4f}')

    # 如果连续几个周期性能没有改善，就提前停止训练
    if no_improvement_count >= early_stopping_patience:
        print(f'Early stopping after {early_stopping_patience} epochs without improvement.')
        break

    # 将预测值和标签添加到DataFrame中
    for slide_id, label, predicted in zip(slide_ids, val_labels, val_scores):
        results_df = results_df.append({'Case_ID': slide_id, 'Label': label, 'Predicted': predicted}, ignore_index=True)

# 关闭TensorBoard写入器
writer.close()

# 保存结果DataFrame为CSV文件
results_df.to_csv('predictions_results_for_auc.csv', index=False)
