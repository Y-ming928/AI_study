import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import matplotlib.pyplot as plt  # 导入matplotlib用于画图
import copy  # 导入copy模块用于深拷贝模型权重

class Trainer:  # 定义Trainer训练器类
    def __init__(
        self, 
        model, 
        trainloader, 
        valloader, 
        lr=0.01, 
        momentum=0.9, 
        device=None,
        early_stop=False,            # 是否启用早停，默认为False
        early_stop_mode='val_loss',  # 早停的模式，可选'val_loss'或'val_acc'
        patience=5,                  # 早停的容忍轮数
        save_path='best_model.pth'   # 保存最佳模型的路径
    ):
        self.model = model  # 保存模型
        self.trainloader = trainloader  # 保存训练数据加载器
        self.valloader = valloader  # 保存验证数据加载器
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定设备，优先GPU
        self.criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)  # 初始化SGD优化器
        self.model.to(self.device)  # 将模型转移到指定设备

        # 用于记录损失与准确率，便于后续可视化
        self.train_losses = []  # 记录每个epoch训练损失
        self.val_losses = []  # 记录每个epoch验证损失
        self.train_accuracies = []  # 记录每个epoch训练准确率
        self.val_accuracies = []  # 记录每个epoch验证准确率

        # 早停与模型保存相关参数
        self.early_stop = early_stop  # 是否启用早停
        self.early_stop_mode = early_stop_mode  # 早停指标类型
        self.patience = patience  # 早停的容忍度
        self.save_path = save_path  # 最佳模型存储路径

    def evaluating(self, dataloader):  # 定义评估方法，输入dataloader
        self.model.eval()  # 转为评估模式
        correct = 0  # 统计正确预测个数
        total = 0  # 统计总样本数
        running_loss = 0.0  # 累积损失初始化
        with torch.no_grad():  # 禁用梯度以节省资源
            for images, labels in dataloader:  # 遍历数据
                images = images.to(self.device)  # 图像移到设备
                labels = labels.to(self.device)  # 标签移到设备
                outputs = self.model(images)  # 前向传播
                loss = self.criterion(outputs, labels)  # 计算损失
                running_loss += loss.item()  # 累加损失
                predicted = torch.argmax(outputs, dim=1)  # 得到预测类别
                total += labels.size(0)  # 累加样本总数
                correct += (predicted == labels).sum().item()  # 累加正确数
        acc = 100.0 * correct / total  # 计算准确率
        avg_loss = running_loss / len(dataloader)  # 平均损失
        return acc, avg_loss  # 返回准确率和平均损失

    def train(self, epochs=10):  # 定义训练方法，默认训练10轮
        # 早停相关的初始化
        best_metric = None  # 最佳指标初始化
        best_epoch = 0  # 最佳权重所在epoch
        best_model_wts = copy.deepcopy(self.model.state_dict())  # 深拷贝初始模型参数
        wait = 0  # 没有提升的轮数计数器

        for epoch in range(epochs):  # 遍历每个epoch
            self.model.train()  # 切换到训练模式
            running_loss = 0.0  # 初始化本轮损失累加
            for batch_idx, (images, labels) in enumerate(self.trainloader):  # 遍历每个小批量
                images = images.to(self.device)  # 图像移至设备
                labels = labels.to(self.device)  # 标签移至设备
                self.optimizer.zero_grad()  # 梯度清零
                outputs = self.model(images)  # 前向传播
                loss = self.criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 优化器更新参数
                running_loss += loss.item()  # 累加损失
                if (batch_idx + 1) % 100 == 0:  # 每100个batch打印一次日志
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')  # 打印当前Loss
            avg_train_loss = running_loss / len(self.trainloader)  # 计算平均训练损失
            train_acc, _ = self.evaluating(self.trainloader)  # 计算训练集的准确率
            val_acc, avg_val_loss = self.evaluating(self.valloader)  # 计算验证集准确率和损失

            # 记录日志，用于可视化
            self.train_losses.append(avg_train_loss)  # 记录训练损失
            self.val_losses.append(avg_val_loss)  # 记录验证损失
            self.train_accuracies.append(train_acc)  # 记录训练准确率
            self.val_accuracies.append(val_acc)  # 记录验证准确率

            # 打印本轮训练、验证各项指标
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # ----------- 早停与模型保存逻辑开始 -----------
            stop_now = False  # 标志是否需要提前终止训练
            metric = None  # 当前轮待比较的指标

            if self.early_stop:  # 若启动早停
                if self.early_stop_mode == 'val_loss':  # 按验证损失早停
                    metric = avg_val_loss
                    compare = (lambda a, b: a < b)  # 损失越低越好
                elif self.early_stop_mode == 'val_acc':  # 按验证准确率早停
                    metric = val_acc
                    compare = (lambda a, b: a > b)  # 准确率越高越好
                else:
                    raise ValueError("early_stop_mode must be 'val_loss' or 'val_acc'")  # 参数不合法时报错

                if best_metric is None or compare(metric, best_metric):  # 新的指标更优
                    best_metric = metric  # 更新最佳指标
                    best_epoch = epoch + 1  # 记录最佳epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())  # 保存最佳模型权重
                    wait = 0  # 重置无提升次数
                    torch.save(self.model.state_dict(), self.save_path)  # 保存模型至文件
                    print(f"Best model saved at epoch {epoch+1}.")  # 打印保存提示
                else:  # 没有提升
                    wait += 1  # 无提升次数加1
                    print(f'No improvement. Early stop counter: {wait}/{self.patience}')  # 打印早停计数器
                    if wait >= self.patience:  # 超过容忍度
                        print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch} with best_metric={best_metric:.4f}")
                        stop_now = True  # 提前终止训练

            if stop_now:  # 如果需要提前终止
                break  # 跳出循环

        # 恢复到最佳的模型权重
        if self.early_stop and best_metric is not None:  # 如有早停且有最佳指标
            print(f'Loading best model weights from epoch {best_epoch}')  # 打印提示
            self.model.load_state_dict(best_model_wts)  # 恢复权重

    # 增加回归评估方法
    def regression_evaluating(self, dataloader):
        """评估回归任务，仅返回平均损失（如均方误差），与分类准确率无关"""
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        return avg_loss

    # 增加回归任务的训练方法
    def train_regression(self, epochs=10):
        """
        针对回归任务的训练方法, 会记录train_losses, val_losses，不涉及准确率
        """
        best_metric = None
        best_epoch = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        wait = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')
            avg_train_loss = running_loss / len(self.trainloader)
            avg_val_loss = self.regression_evaluating(self.valloader)

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            stop_now = False
            metric = None

            if self.early_stop:
                if self.early_stop_mode == 'val_loss':
                    metric = avg_val_loss
                    compare = (lambda a, b: a < b)
                else:
                    raise ValueError("For regression, early_stop_mode must be 'val_loss'")

                if best_metric is None or compare(metric, best_metric):
                    best_metric = metric
                    best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    wait = 0
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Best regression model saved at epoch {epoch+1}.")
                else:
                    wait += 1
                    print(f'No improvement. Early stop counter: {wait}/{self.patience}')
                    if wait >= self.patience:
                        print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch} with best_metric={best_metric:.4f}")
                        stop_now = True

            if stop_now:
                break

        if self.early_stop and best_metric is not None:
            print(f'Loading best regression model weights from epoch {best_epoch}')
            self.model.load_state_dict(best_model_wts)

    # plot_metrics支持acc和回归情形，缺省参数 acc=True
    def plot_metrics(self, acc=True):
        """
        绘制损失和准确率/回归损失曲线
        :param acc: 是否绘制准确率曲线，回归任务建议传入False
        """
        epochs = range(1, len(self.train_losses) + 1)
        if acc:
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(epochs, self.train_losses, label='Train Loss')
            plt.plot(epochs, self.val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(epochs, self.train_accuracies, label='Train Acc')
            plt.plot(epochs, self.val_accuracies, label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy Curve')
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            # 只画损失（回归）
            plt.figure(figsize=(6,5))
            plt.plot(epochs, self.train_losses, label='Train Loss')
            plt.plot(epochs, self.val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve (Regression)')
            plt.legend()
            plt.tight_layout()
            plt.show()