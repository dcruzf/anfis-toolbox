# ANFIS Toolbox - Sistema de Logging

O ANFIS Toolbox usa o módulo `logging` do Python para controlar a saída de informações durante o treinamento e operações, seguindo as melhores práticas de desenvolvimento de bibliotecas Python.

## Configuração Rápida

### Habilitar logs de treinamento (formato simples)
```python
from anfis_toolbox import enable_training_logs

enable_training_logs()
# Agora os logs de treinamento aparecerão no console
```

### Desabilitar logs de treinamento
```python
from anfis_toolbox import disable_training_logs

disable_training_logs()
# Os logs de treinamento não aparecerão mais
```

### Configuração personalizada
```python
from anfis_toolbox import setup_logging

# Configurar nível de debug
setup_logging(level="DEBUG")

# Configurar formato personalizado
setup_logging(
    level="INFO",
    format_string="[%(levelname)s] %(name)s: %(message)s"
)
```

## Uso Básico

```python
import numpy as np
from anfis_toolbox import ANFIS, GaussianMF, enable_training_logs

# Configurar logging
enable_training_logs()

# Criar modelo ANFIS
input_mfs = {
    'x1': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    'x2': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
}
model = ANFIS(input_mfs)

# Dados de treinamento
x_train = np.random.randn(100, 2)
y_train = np.sum(x_train, axis=1, keepdims=True)

# Treinar com logs habilitados
losses = model.fit(x_train, y_train, epochs=50, verbose=True)
# Saída: Epoch 5/50, Loss: 0.123456
#        Epoch 10/50, Loss: 0.098765
#        ...
```

## Controle de Verbosidade

O parâmetro `verbose` nos métodos de treinamento controla se as mensagens de progresso são enviadas para o sistema de logging:

```python
# Com logs de progresso
losses = model.fit(x_train, y_train, verbose=True)

# Sem logs de progresso
losses = model.fit(x_train, y_train, verbose=False)
```

## Níveis de Logging

O sistema suporta todos os níveis padrão do Python:

- `DEBUG`: Informações detalhadas para depuração
- `INFO`: Informações gerais (logs de treinamento)
- `WARNING`: Avisos (padrão quando logs são desabilitados)
- `ERROR`: Erros
- `CRITICAL`: Erros críticos

## Vantagens sobre `print()`

1. **Padrão da Indústria**: Usa o sistema de logging do Python, amplamente aceito
2. **Configurável**: Usuários podem controlar facilmente o que querem ver
3. **Flexível**: Diferentes níveis de verbosidade
4. **Compatível**: Funciona bem com outros sistemas de logging
5. **Não Intrusivo**: Bibliotecas não devem imprimir diretamente no console
6. **Testável**: Logs podem ser facilmente desabilitados em testes

## Configuração para Diferentes Cenários

### Para Desenvolvimento/Depuração
```python
from anfis_toolbox import setup_logging

setup_logging(level="DEBUG")
```

### Para Produção
```python
from anfis_toolbox import disable_training_logs

disable_training_logs()  # Só mostra warnings e erros
```

### Para Jupyter Notebooks
```python
from anfis_toolbox import enable_training_logs

enable_training_logs()  # Formato simples, ideal para notebooks
```

### Em Testes Automatizados
```python
import logging
from anfis_toolbox import setup_logging

# Desabilitar completamente ou só críticos
setup_logging(level="CRITICAL")
```

## Compatibilidade com Ruff

Este sistema resolve a violação da regra T201 do Ruff, que proíbe o uso de `print()` statements em bibliotecas, seguindo as melhores práticas de desenvolvimento Python.
