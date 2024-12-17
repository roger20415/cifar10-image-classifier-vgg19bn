class ValidationMonitor:
    def __init__(self):
        self._val_loss: list[float] = []
        self._val_acc: list[float] = []
        
        self._running_val_loss: float = 0.0
        self._correct_val_num: int = 0
        self._total_val_num: int = 0
    
    def epoch_reset(self) -> None:
        self._running_val_loss = 0.0
        self._correct_val_num = 0
        self._total_val_num = 0
    
    def add_running_val_loss(self, loss: float) -> None:
        self._running_val_loss += loss
    
    def add_correct_val_num(self, correct_val_num: int) -> None:
        self._correct_val_num += correct_val_num
    
    def add_total_val_num(self, total_val_num: int) -> None:
        self._total_val_num += total_val_num
    
    def append_val_loss(self, batch_num: int) -> None:
        self._val_loss.append(self._running_val_loss / batch_num)
    
    def append_val_acc(self) -> None:
        self._val_acc.append(100 * self._correct_val_num / self._total_val_num)
        
    def get_last_val_acc(self) -> float:
        return self._val_acc[-1]
    
    def get_val_loss(self) -> list[float]:
        return self._val_loss
    
    def get_val_acc(self) -> list[float]:
        return self._val_acc