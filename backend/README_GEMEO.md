# Sentinela · demonstração de gêmeo digital

Em uma máquina fictícia (MAQ-01), o painel apresenta a última leitura,
o score da Random Forest e o histórico das condições e dos alertas.
Cada nova leitura chama o mesmo `POST /predict` que serve o frontend.
O estado e o histórico desta demonstração ficam na página aberta.

## Abrir a demonstração

Coloque `ai4i2020.csv` ao lado da pasta `backend/`. Você também pode
configurar `SENTINELA_DATASET` com o caminho do CSV.

Em `backend/`, use o ambiente Python 3.13 descrito no README principal.
No PowerShell, com as dependências instaladas:

```powershell
..\.venv\Scripts\python.exe -m uvicorn gemeo:app --host 127.0.0.1 --port 8000
```

Abra [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo).
Se a API já estiver rodando nessa porta, encerre-a antes ou use outra
porta. No Linux ou macOS, use `../.venv/bin/python`.

## Roteiro para apresentar

1. A primeira leitura aparece automaticamente. Aponte as medições,
   o score e o limiar de alerta de 0,50.
2. Clique em **Reproduzir** ou avance as leituras uma por vez. Na quinta
   leitura (UDI 4131), o score sobe para aproximadamente 0,873 e aparece
   a indicação de inspeção.
3. Pause e consulte a curva e a tabela. Reinicie para repetir o cenário.

O replay usa 80 registros consecutivos do AI4I, ordenados por UDI. A
sequência foi escolhida para demonstrar uma mudança de estado e um alerta.
O ID da máquina e os horários são simulados. Os códigos de falha do CSV
ficam fora da inferência. A demonstração ilustra um fluxo de monitoramento
e não mede desempenho em teste. Para conectar uma máquina real, seria
preciso substituir o replay por uma fonte de medições identificadas e
validar o modelo com dados da operação.

[Fonte dos dados: UCI AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i)
