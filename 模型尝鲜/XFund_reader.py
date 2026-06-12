#!/usr/bin/env python
# coding=utf-8
"""
独立的 XFund 数据集加载脚本
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class XFundDataset(Dataset):
    """
    独立的 XFund 数据集类
    """
    
    # 标签映射
    LABEL2ID = {
        "O": 0,
        'B-HEADER': 1,
        'I-HEADER': 2,
        'B-QUESTION': 3,
        'I-QUESTION': 4,
        'B-ANSWER': 5,
        'I-ANSWER': 6,
    }
    
    def __init__(self, data_dir, tokenizer, mode='train', max_length=512):
        """
        初始化 XFund 数据集
        
        Args:
            data_dir: 数据目录路径
            tokenizer: 分词器
            mode: 模式 ('train', 'val')
            max_length: 最大序列长度
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
        # 加载数据
        self.load_data()
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_data(self):
        """加载和预处理数据"""
        # 构建文件路径
        filename = f"{self.mode}.json"
        file_path = os.path.join(self.data_dir, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data_file = json.load(f)
        
        # 处理数据
        self.processed_data = []
        
        for doc in data_file['documents']:
            width, height = doc['img']['width'], doc['img']['height']
            
            # 处理文档中的每个元素
            for item in doc['document']:
                text = item['text']
                box = item['box']
                label = item['label']
                
                # 归一化边界框
                normalized_box = self.normalize_box(box, width, height)
                
                # 分词
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                
                # 创建标签
                labels = self.create_labels(label, len(token_ids))
                
                # 添加到处理后的数据中
                self.processed_data.append({
                    'text': text,
                    'token_ids': token_ids,
                    'bbox': normalized_box,
                    'labels': labels,
                    'image_path': os.path.join(self.data_dir, 'images', doc['img']['fname'])
                })
    
    def normalize_box(self, box, width, height):
        """归一化边界框"""
        x0, y0, x1, y1 = box
        return [
            int((x0 / width) * 1000),
            int((y0 / height) * 1000),
            int((x1 / width) * 1000),
            int((y1 / height) * 1000)
        ]
    
    def create_labels(self, label, length):
        """创建标签序列"""
        if label == 'OTHER':
            return [self.LABEL2ID["O"]] * length
        else:
            labels = [self.LABEL2ID[f'B-{label}']] + [self.LABEL2ID[f'I-{label}']] * (length - 1)
            return labels
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.processed_data[idx]
        
        # 处理文本
        input_ids = [self.tokenizer.cls_token_id] + item['token_ids'] + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        # 处理标签
        labels = [-100] + item['labels'] + [-100]
        
        # 处理边界框
        bbox = [[0, 0, 0, 0]] + [item['bbox']] * len(item['token_ids']) + [[1000, 1000, 1000, 1000]]
        
        # 处理图像
        image_path = item['image_path']
        img = Image.open(image_path).convert('RGB')
        image = self.image_transform(img)
        
        # 确保长度不超过最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            bbox = bbox[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'bbox': bbox,
            'images': image
        }

def main():
    """主函数"""
    # 配置参数
    data_dir = 'path/to/xfund/data'  # 替换为您的数据路径
    model_name = 'your_model_name'  # 替换为您的模型名称
    
    # 加载分词器
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建数据集
    dataset = XFundDataset(data_dir=data_dir, tokenizer=tokenizer, mode='train')
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 预览数据
    print(f"数据集大小: {len(dataset)}")
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  BBoxes shape: {batch['bbox'].shape}")
        print(f"  Images shape: {batch['images'].shape}")
        
        if i >= 2:  # 只显示前2个批次
            break

if __name__ == "__main__":
    main()































