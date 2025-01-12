class TrainingMonitor:
    def __init__(self) -> None:
        self._train_loss: list[float] = []
        self._train_acc: list[float] = []
        
        self._running_loss: float = 0.0
        self._correct_train_num: int = 0
        self._total_train_num: int = 0
        
    def epoch_reset(self) -> None:
        self._running_loss = 0.0
        self._correct_train_num = 0
        self._total_train_num = 0
        
    def add_running_loss(self, loss: float) -> None:
        self._running_loss += loss     
        
    def add_correct_train_num(self, correct_train_num: int) -> None:
        self._correct_train_num += correct_train_num
    
    def add_total_train_num(self, total_train_num: int) -> None:
        self._total_train_num += total_train_num
    
    def append_train_loss(self, batch_num: int) -> None:
        self._train_loss.append(self._running_loss / batch_num)
    
    def append_train_acc(self) -> None:
        self._train_acc.append(100 * self._correct_train_num / self._total_train_num)
    
    def get_train_loss(self) -> list[float]:
        return self._train_loss

    def get_train_acc(self) -> list[float]:
        return self._train_acc