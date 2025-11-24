```bash
py -m venv venv              # установка интерпретатора 
.\venv\Scripts\Activate.ps1  # для виртуальной среды
```

```python
pip install -r requirements.txt  # установка зависимостей проекта 
```

Опционально
```bash
nvidia-smi  # В выводе будет строка: CUDA Version: *.*
```

```python
py test.py  # покажет вашу информацию о cuda
```
![cuda](./images/cuda.png)

При необходимости
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```



![metrics_10epoch](./images/summary.png)
